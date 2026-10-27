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
6. **Serve predictions** — `model/predict.py` loads that saved model when the API starts. A client sends 30 measurements and an API key to `POST /predict`; the API returns the predicted class. Health endpoints are available separately and do not require authentication.

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

Configure an API key and start the production WSGI server from the `model` directory:

```powershell
$env:API_KEY = ([guid]::NewGuid().ToString("N") + [guid]::NewGuid().ToString("N"))
python -c "from waitress import serve; from predict import app; serve(app, host='0.0.0.0', port=5000, threads=4)"
```

The API listens on `http://127.0.0.1:5000`.

| Method | Route | Authentication | Purpose |
|---|---|---|---|
| `GET` | `/health/live` | None | Process liveness check |
| `GET` | `/health/ready` | None | Readiness check after loading the model |
| `GET` | `/metrics` | None | Prometheus request metrics |
| `POST` | `/predict` | `X-API-Key` header | Predict one class from 30 numeric features |

`/predict` validates the JSON body, feature count and numeric finite values. Requests without the configured key return `401`; requests are rate limited to 30 per minute by default. To try it locally, send the same API key in the `X-API-Key` header with the 30 input values in the dataset's original order.

Run the API tests from the `model` directory with `API_KEY` set:

```powershell
$env:API_KEY = "test-api-key-that-is-at-least-32-characters"
python -m unittest discover -s tests -v
```

## Deployment and monitoring context

The Dockerfile packages the API with Waitress, runs it as a non-root user, and includes a readiness health check. The original GitHub Actions workflow trains the model and builds/pushes a Docker image to Docker Hub when code is pushed to `main` (Docker Hub credentials must be configured as repository secrets).

The API exports Prometheus request metrics at `/metrics`, but the repository's Kubernetes, Prometheus, and Grafana deployment files still need configuration before they can be used as a monitoring stack. The separate `model/drift_check.py` script demonstrates an offline comparison of dataset subsets; it does not detect live feature drift.

## Deploy to AWS (ECR + ECS Fargate)

The project can run on **Amazon ECS Fargate**, with **Amazon ECR** storing the container image. This serves the API; it does not run the Kubeflow pipeline. AWS resources can incur charges; review pricing and delete resources when finished.

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

1. In the AWS Console, open **IAM → Roles** and create (or confirm) the ECS task execution role named `ecsTaskExecutionRole`, using the **Elastic Container Service Task** trusted entity and attaching the AWS-managed `AmazonECSTaskExecutionRolePolicy`. ECS uses this role to pull the private ECR image and write container logs. Add `secretsmanager:GetSecretValue` permission scoped to the specific API-key secret ARN (and `kms:Decrypt` for its customer-managed KMS key, if applicable).
2. Open **Amazon ECS → Clusters → Create cluster**. Choose the **AWS Fargate (serverless)** infrastructure option, give the cluster a name such as `breast-cancer-cluster`, and create it.

Do not put AWS credentials inside the image or application. The app itself does not need an AWS task role for its current inference behavior.

### 4. Create a Fargate task definition

In **ECS → Task definitions → Create new task definition**:

1. Choose **AWS Fargate**, Linux/X86_64, and a task size such as **1 vCPU and 2 GB memory** for a small demo.
2. Select `ecsTaskExecutionRole` as the **Task execution role**.
3. Add a container named `breast-cancer-api` and set its image to the ECR URI with the tag you pushed, for example `<account-id>.dkr.ecr.us-east-1.amazonaws.com/breast-cancer-api:v1`.
4. Configure a container port mapping for **TCP 5000**.
5. Enable the `awslogs` log driver and select/create a CloudWatch Logs group (for example `/ecs/breast-cancer-api`) in the same Region.
6. Configure a container health check with command `python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health/ready', timeout=3)"`, interval 30 seconds, timeout 5 seconds, retries 3, and start period 30 seconds. Configure this in the task definition so ECS uses it for service health.
7. Save the task definition.

The image starts Waitress and the API listens on port `5000`. The app requires an `API_KEY` at startup; create a secret in AWS Secrets Manager and configure the task definition's container `secrets` entry to inject it as `API_KEY`. Do not store the API key as plaintext in the task definition or GitHub.

### 5. Run the service

In the cluster, choose **Create service** and configure:

1. Compute option **Launch type → Fargate** and select the task definition revision you created.
2. Set the desired task count to `1`.
3. Choose a VPC and public subnet(s). For a temporary demo without a load balancer, enable **Auto-assign public IP**.
4. For a short-lived direct-IP test, create or select a security group whose inbound rule allows **TCP 5000 only from your current public IP**. Do not open port 5000 to `0.0.0.0/0`. For regular access, prefer the ALB setup below.
5. Create the service and wait until the task is `RUNNING`.

For a direct-IP test, open the running task's **Networking** details, copy its public IP, and test:

```powershell
$TaskUrl = "http://<task-public-ip>:5000"
curl.exe "$TaskUrl/health/ready"
```

Expected health response:

```json
{"status":"ready"}
```

To try inference, send the 30 numeric dataset features in the same order used by scikit-learn:

```powershell
curl.exe -X POST "$TaskUrl/predict" `
  -H "Content-Type: application/json" `
  -H "X-API-Key: <your-api-key>" `
  -d '{"features":[17.99,10.38,122.8,1001.0,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019.0,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]}'
```

The direct task IP can change when ECS replaces or restarts a task. For a stable endpoint, use an **Application Load Balancer (ALB)**: create an internet-facing ALB in at least two public subnets, configure an HTTPS listener and a target group of type **IP** on port `5000` with health-check path `/health/ready`, then attach it to the ECS service. Allow inbound `443` on the ALB security group, and allow port `5000` on the task security group **only from the ALB security group**. Configure a domain and TLS certificate with AWS Certificate Manager; redirect HTTP to HTTPS if you also configure port 80. Enable `TRUST_PROXY=1` in the task only when the task security group ensures all client traffic comes through the ALB.

### 6. Updates, logs, and cleanup

To release an updated model or API, rebuild and push a new version tag to ECR, update the ECS task definition to reference that tag, and update the service to use the new task definition revision. Check startup/runtime output in the task's CloudWatch log group. If a task fails to start, verify that the image tag exists in ECR, `model/model.pkl` was included in the image, the execution role can pull from ECR, and the task can reach required AWS endpoints.

When finished, delete the ECS service (and ALB/target group if created), then the cluster and ECR repository if no longer needed. Also remove unused CloudWatch log groups. ECS tasks, load balancers, image storage, and logs may incur charges.

### Automated AWS deployment from GitHub Actions

`.github/workflows/deploy-aws.yml` tests the API, trains the model, builds a uniquely tagged image, pushes it to ECR, registers a new ECS task-definition revision, and waits for the ECS service to become stable. It uses GitHub's OIDC federation to assume an AWS role; no long-lived AWS access key is stored in GitHub. The job is disabled until explicitly enabled.

Before enabling it:

1. Create the ECR repository, ECS cluster, Fargate service, task definition, and Secrets Manager API-key secret using the manual steps above.
2. Create an IAM OIDC identity provider for `token.actions.githubusercontent.com` in the AWS account (if one does not already exist).
3. Create a GitHub Actions IAM role with a trust policy restricted to audience `sts.amazonaws.com` and this repository's `main` branch subject: `repo:Ayush-Singh986/MLOPS-endtoend-pipeline:ref:refs/heads/main`. Replace `<account-id>` with the AWS account ID:

   ```json
   {
     "Version": "2012-10-17",
     "Statement": [{
       "Effect": "Allow",
       "Principal": {
         "Federated": "arn:aws:iam::<account-id>:oidc-provider/token.actions.githubusercontent.com"
       },
       "Action": "sts:AssumeRoleWithWebIdentity",
       "Condition": {
         "StringEquals": {
           "token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
           "token.actions.githubusercontent.com:sub": "repo:Ayush-Singh986/MLOPS-endtoend-pipeline:ref:refs/heads/main"
         }
       }
     }]
   }
   ```

4. Attach a least-privilege permissions policy to that role. It needs `ecr:GetAuthorizationToken` on `*`; ECR image upload actions only on the chosen repository ARN; ECS describe/register actions for the chosen task definition (some describe/register actions require `*`); `ecs:UpdateService` and `ecs:DescribeServices` only on the chosen service ARN; and `iam:PassRole` only for the ECS task execution/task roles referenced by the task definition, conditioned on `iam:PassedToService` being `ecs-tasks.amazonaws.com`. Avoid `AdministratorAccess`. Confirm the final resource scopes against the ECS/ECR IAM action reference for your account.
5. In GitHub **Settings → Secrets and variables → Actions → Variables**, add:

   | Variable | Example | Purpose |
   |---|---|---|
   | `AWS_DEPLOY_ENABLED` | `true` | Enables the deployment job |
   | `AWS_REGION` | `us-east-1` | AWS Region for ECR/ECS |
   | `AWS_ROLE_TO_ASSUME` | `arn:aws:iam::123456789012:role/github-ecs-deploy` | OIDC role ARN |
   | `ECR_REPOSITORY` | `breast-cancer-api` | Existing ECR repository name |
   | `ECS_CLUSTER` | `breast-cancer-cluster` | ECS cluster name |
   | `ECS_SERVICE` | `breast-cancer-api` | Existing ECS service name |
   | `ECS_TASK_DEFINITION` | `breast-cancer-api` | Task-definition family or ARN |
   | `ECS_CONTAINER_NAME` | `breast-cancer-api` | Container name inside task definition |

6. Confirm the ECS task definition injects `API_KEY` from Secrets Manager, uses port `5000`, and has the health check enabled. Configure an ALB for HTTPS and restrict the task security group to ALB traffic for a public service.
7. Push a change to `main` that touches `model/` or `docker/`, or manually run **Actions → Train and deploy to AWS ECS → Run workflow**.

The workflow updates only the container image in the existing task definition. It does not create AWS infrastructure, generate API keys, alter networking, or provision an ALB. Review the AWS role's trust and permissions and the ECS service's network settings before enabling automatic deployments.

### Security and operational limitations

The API now uses Waitress, requires an `API_KEY`, validates prediction inputs, rate-limits requests, logs request metadata/errors without logging feature payloads, and exposes liveness/readiness and Prometheus metrics endpoints. Restrict `/metrics` to trusted monitoring systems. Rate limiting uses in-memory storage by default; for multiple ECS tasks, configure a shared Redis-compatible `RATELIMIT_STORAGE_URI` and private networking. Terminate HTTPS at an ALB with ACM; do not expose the task directly to the internet. This remains a learning example, not a production-ready medical service. Do not send real patient data to it or use its predictions for clinical decisions.

> **Disclaimer:** This is an educational example, not a medical diagnostic tool. Do not use its predictions for clinical decisions.
