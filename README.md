# Breast Cancer MLOps Pipeline

This project demonstrates how a machine-learning model can move from data and training to an API that serves predictions. It uses scikit-learn's Breast Cancer Wisconsin dataset to classify tumors as malignant (`0`) or benign (`1`).

**Repository owner:** [Ayush-Singh986](https://github.com/Ayush-Singh986)

## Architecture

![MLOps pipeline architecture](architecture.png)

## Project walkthrough

### Runnable local workflow

1. **Load the data** — `model/train.py` loads the breast-cancer dataset directly from scikit-learn. Each record has 30 numeric measurements and a binary target.
2. **Split into train and test sets** — The script uses an 80/20 split with a fixed random seed, making the experiment repeatable.
3. **Train candidate models** — It fits Random Forest, Logistic Regression, and XGBoost classifiers using the same training data.
4. **Evaluate and track experiments** — For each model, it calculates accuracy, precision, recall, and F1 score on the test set. MLflow records these metrics, the model name, and the trained model artifact in `model/mlruns`.
5. **Choose and save a model** — The script selects the model with the highest test accuracy and saves it to `model/model.pkl`.
6. **Serve predictions** — `model/predict.py` loads that saved model when the Flask app starts. A client sends the 30 measurements as JSON to `POST /predict`; the API returns the predicted class. `GET /` is a basic health check.

### Extended Kubeflow workflow

`model/kubeflow_code.py` describes a separate, more detailed pipeline intended to run in Kubeflow Pipelines:

1. **Ingest and validate** — Load the dataset and check that it is non-empty, contains a target column, and has both classes.
2. **Engineer features** — Add a small set of statistical and ratio-based features.
3. **Split and preprocess** — Make stratified training, calibration, and test splits (60/20/20). Fit preprocessing on the training data and apply it to the other splits.
4. **Train five candidates** — Compare standard XGBoost, cost-sensitive XGBoost, XGBoost with SMOTE, cost-sensitive LightGBM, and a feature-selected Random Forest.
5. **Evaluate and select** — Calculate classification metrics, AUC-ROC, false positives/negatives, and business cost. Select by highest recall, then lowest business cost, then highest AUC-ROC.
6. **Calibrate predictions** — Use the calibration split for conformal calibration, then evaluate prediction-set coverage and size on the test split.
7. **Apply deployment gates** — Continue only when recall is at least `0.80`, AUC-ROC at least `0.85`, and coverage at least `0.90`, followed by model size and latency checks.

The compiled Kubeflow specification is `model/kubeflowpipeline.yaml`. This workflow is distinct from the local MLflow training script. Its final deployment component is currently a stub: it logs the model artifact path but does not deploy the model.

## Run the local workflow

Requires Python 3.11. From the repository root, create an environment and install dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r model\requirements.txt
```

Train models and save the selected model:

```powershell
Set-Location model
python train.py
```

View the MLflow runs (from the `model` directory):

```powershell
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5001
```

Open <http://127.0.0.1:5001> to compare the recorded metrics and model runs.

Start the API in a separate terminal, from the repository root:

```powershell
python model\predict.py
```

The API listens on `http://127.0.0.1:5000`. Send a JSON object with the 30 numeric features, in the dataset's original order, to `POST /predict`. The response is a class prediction, such as `{"prediction": 0}`.

## Deployment and monitoring context

The Dockerfile packages the Flask API, and the GitHub Actions workflow trains the model and builds/pushes a Docker image when code is pushed to `main` (Docker Hub credentials must be configured as repository secrets). The workflow does not deploy to Kubernetes.

Kubernetes, Prometheus, and Grafana assets are included as deployment scaffolding, but are not fully connected to the API. In particular, some configuration files are empty and the API does not expose the `/metrics` endpoint expected by the ServiceMonitor. The separate `model/drift_check.py` script demonstrates an offline comparison of two dataset subsets; it does not monitor live API traffic.

## Deploy to AWS (ECR + ECS Fargate)

The following is a manual learning/development deployment using **Amazon ECR** to store the Docker image and **Amazon ECS on AWS Fargate** to run it. It deploys the existing Flask API, not the Kubeflow pipeline. AWS resources can incur charges; review pricing and delete resources when finished.

### 1. Prerequisites

- An AWS account and AWS CLI v2 configured with credentials for creating ECR/ECS resources (`aws configure`).
- Docker Desktop running with Linux containers enabled.
- Python 3.11 and the project dependencies installed.
- An AWS Region selected for all resources, such as `us-east-1`.

From the repository root, train the model before building the image. Training writes the selected model to `model/model.pkl`, which the Docker image needs at startup:

```powershell
Set-Location model
python train.py
Set-Location ..
```

### 2. Create an ECR repository and push the image

In PowerShell at the repository root, set the Region, then use AWS CLI to create a private ECR repository and retrieve your account ID:

```powershell
$Region = "us-east-1"
$AccountId = aws sts get-caller-identity --query Account --output text
$Repository = "breast-cancer-api"
$ImageTag = "v1"
$EcrUri = "$AccountId.dkr.ecr.$Region.amazonaws.com/$Repository"

aws ecr create-repository --repository-name $Repository --region $Region
```

If the repository already exists, the create command reports that; continue with the existing repository. Authenticate Docker, build the image from the repository root (the Dockerfile expects `model/` in the build context), then tag and push it:

```powershell
aws ecr get-login-password --region $Region |
  docker login --username AWS --password-stdin "$AccountId.dkr.ecr.$Region.amazonaws.com"

docker build --platform linux/amd64 -f docker\Dockerfile -t "${Repository}:$ImageTag" .
docker tag "${Repository}:$ImageTag" "${EcrUri}:$ImageTag"
docker push "${EcrUri}:$ImageTag"
```

Use an immutable release tag such as `v1` rather than `latest` so task definitions can point to a specific image version.

### 3. Prepare ECS permissions and cluster

1. In the AWS Console, open **IAM → Roles** and create (or confirm) the ECS task execution role named `ecsTaskExecutionRole`, using the **Elastic Container Service Task** trusted entity and attaching the AWS-managed `AmazonECSTaskExecutionRolePolicy`. ECS uses this role to pull the private ECR image and write container logs.
2. Open **Amazon ECS → Clusters → Create cluster**. Choose the **AWS Fargate (serverless)** infrastructure option, give the cluster a name such as `breast-cancer-cluster`, and create it.

Do not put AWS credentials inside the image or application. The app itself does not need an AWS task role for its current inference behavior.

### 4. Create a Fargate task definition

In **ECS → Task definitions → Create new task definition**:

1. Choose **AWS Fargate**, Linux/X86_64, and a task size such as **1 vCPU and 2 GB memory** for a small demo.
2. Select `ecsTaskExecutionRole` as the **Task execution role**.
3. Add a container named `breast-cancer-api` and set its image to the ECR URI with the tag you pushed, for example `<account-id>.dkr.ecr.us-east-1.amazonaws.com/breast-cancer-api:v1`.
4. Configure a container port mapping for **TCP 5000**.
5. Enable the `awslogs` log driver and select/create a CloudWatch Logs group (for example `/ecs/breast-cancer-api`) in the same Region.
6. Save the task definition.

The image starts `python predict.py` and the Flask app listens on port `5000`.

### 5. Run the service

In the cluster, choose **Create service** and configure:

1. Compute option **Launch type → Fargate** and select the task definition revision you created.
2. Set the desired task count to `1`.
3. Choose a VPC and public subnet(s). For a temporary demo without a load balancer, enable **Auto-assign public IP**.
4. Create or select a security group whose inbound rule allows **TCP 5000 only from your current public IP** (replace the example `203.0.113.10/32` with your own IP in CIDR notation). Do not open port 5000 to `0.0.0.0/0` for an internet-facing service.
5. Create the service and wait until the task is `RUNNING`.

For this direct-IP demo, open the running task's **Networking** details, copy its public IP, and test (replace the example address with the task's actual public IP):

```powershell
$TaskIp = "54.12.34.56"
curl.exe "http://${TaskIp}:5000/"
```

Expected health response:

```json
{"status":"model API running"}
```

To try inference, send the 30 numeric dataset features in the same order used by scikit-learn:

```powershell
curl.exe -X POST "http://${TaskIp}:5000/predict" `
  -H "Content-Type: application/json" `
  -d '{"features":[17.99,10.38,122.8,1001.0,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019.0,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]}'
```

The direct task IP can change when ECS replaces or restarts a task. For a stable endpoint, use an **Application Load Balancer (ALB)**: create an internet-facing ALB in at least two public subnets, configure an HTTP/HTTPS listener and a target group of type **IP** on port `5000` with health-check path `/`, then attach it to the ECS service. Allow inbound `80`/`443` on the ALB security group, and allow port `5000` on the task security group **only from the ALB security group**. For a real deployment, configure a domain and TLS certificate with AWS Certificate Manager and use HTTPS.

### 6. Updates, logs, and cleanup

To release an updated model or API, rebuild and push a new version tag to ECR, update the ECS task definition to reference that tag, and update the service to use the new task definition revision. Check startup/runtime output in the task's CloudWatch log group. If a task fails to start, verify that the image tag exists in ECR, `model/model.pkl` was included in the image, the execution role can pull from ECR, and the task can reach required AWS endpoints.

When finished, delete the ECS service (and ALB/target group if created), then the cluster and ECR repository if no longer needed. Also remove unused CloudWatch log groups. ECS tasks, load balancers, image storage, and logs may incur charges.

### Production notes

This deployment is for learning, not a production-ready medical service. `predict.py` currently starts Flask's development server; replace it with a production WSGI server before production use. Add input validation, authentication/authorization, HTTPS, rate limiting, appropriate logging/monitoring, and operational health checks. Do not send real patient data to this demo. The existing GitHub Actions workflow publishes to Docker Hub and does not deploy to AWS; AWS automation requires a separate ECR/ECS workflow and securely configured AWS IAM federation/permissions.

> **Disclaimer:** This is an educational example, not a medical diagnostic tool. Do not use its predictions for clinical decisions.
