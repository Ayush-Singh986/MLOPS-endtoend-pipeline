# Breast Cancer Detection MLOps Pipeline

## What real-world problem this project solves

Breast cancer diagnosis is a critical healthcare problem. Hospitals and research teams need a way to:

- analyze patient data quickly and consistently,
- train and compare machine learning models reliably,
- track experiments to know which model is best,
- deploy the model in a production-ready API,
- monitor the service for failures, traffic, and health,
- keep the system secure and reproducible in real environments.

Many teams fail in practical ML projects because they build a model in a notebook, but do not solve the operational problems that come after:

- no experiment tracking,
- models trained on one machine cannot be reproduced later,
- no API for real-time predictions,
- no monitoring and alerting,
- no deployment workflow,
- no security around the model endpoint,
- no testability or governance for the ML system.

This project addresses those gaps by turning a breast cancer prediction model into a complete MLOps workflow.

## The business and healthcare value

This system demonstrates how to take a medical classification problem and turn it into a safe, monitored, deployable ML service.

In the real world, the value is:

- early detection support for clinicians,
- faster turnaround for patient risk assessment,
- model comparison with experiment tracking,
- deployment of a prediction service that can be integrated into healthcare systems,
- operational observability for uptime, usage, and reliability,
- a blueprint for trustworthy ML deployment in regulated or sensitive domains.

This is not meant to replace medical expertise. It is a technical demonstration of how ML can be operationalized responsibly.

## The solution implemented in this project

This repository builds an end-to-end MLOps pipeline for breast cancer classification using the Wisconsin breast cancer dataset.

### Core solution components

1. Data loading and training
   - The project reads the Breast Cancer Wisconsin dataset.
   - It splits the data into train and test sets.
   - It trains multiple candidate models such as Random Forest, Logistic Regression, and XGBoost.

2. Experiment tracking with MLflow
   - Model metrics are stored and compared in MLflow.
   - This makes the training process reproducible and auditable.
   - Best-performing models can be selected systematically.

3. Model artifact management
   - The best model is saved locally as a serialized file.
   - The model can be reused by an inference API without retraining.

4. Production API with Flask
   - A secure API receives patient feature values.
   - It validates the input and ensures the request format is correct.
   - It authenticates users using API keys.
   - It returns the model prediction.

5. Monitoring and metrics
   - Prometheus metrics are exposed for request counts, health, and operational visibility.
   - Health endpoints show if the service is live and ready.
   - This is useful for production monitoring, alerting, and dashboards.

6. Containerization and deployment-ready setup
   - Docker configuration packages the API.
   - Kubernetes manifests are included for deployment workflows.
   - AWS ECS deployment patterns are described for cloud hosting.

## Why this matters in real life

A model by itself is not a solution. Real organizations need a complete system that includes:

- data quality,
- model validation,
- reproducibility,
- deployment,
- monitoring,
- security,
- governance.

This project demonstrates that full lifecycle in a simplified but realistic way.

## Architecture overview

The repository follows a typical MLOps pipeline:

- Data source: breast cancer dataset
- Training pipeline: feature processing, model training, evaluation
- Experiment tracking: MLflow
- Model registry/artifact store: local MLflow tracking directory
- Inference service: Flask REST API
- Security: API key validation
- Monitoring: Prometheus + Grafana
- Deployment: Docker, Kubernetes, AWS ECS patterns

## What is inside this repository

- `model/train.py` — trains and evaluates models, logs metrics to MLflow
- `model/predict.py` — API service for prediction requests
- `model/tests/test_predict.py` — validation tests for the API
- `monitoring/` — Prometheus and Grafana setup
- `k8s/` — Kubernetes deployment assets
- `docker/` — Docker packaging files
- `architecture.png` — system architecture diagram

## How this project solves the problem in practice

Instead of building a one-off notebook model, the project creates a pipeline that mirrors how real teams work:

1. Train several models on the same dataset.
2. Measure performance in a controlled and consistent way.
3. Log all experiment metadata into MLflow.
4. Save the best-performing model artifact.
5. Load that artifact into a production API.
6. Protect the endpoint with authentication.
7. Expose health and Prometheus metrics.
8. Prepare the service for container and cloud deployment.

This is the key difference between a research prototype and a deployable ML system.

## Example real-world scenario

A hospital or healthcare analytics team may want to support early cancer risk assessment using structured clinical measurements. A model trained on historical patient data can help flag suspicious cases for specialist review.

This project demonstrates the engineering pattern behind that idea:

- model training pipeline,
- reproducible evaluation,
- deployment as an API,
- monitoring and security,
- the ability to scale to production usage.

## Using the real Kaggle breast cancer dataset

If you provide the real Kaggle breast cancer dataset, this workflow can be adapted directly.

In practice, the training script would change from using the built-in `sklearn.datasets.load_breast_cancer()` function to reading a CSV file such as:

- `data.csv`
- `breast_cancer.csv`
- a Kaggle exported dataset with columns like `diagnosis`, `radius_mean`, `texture_mean`, etc.

Then the workflow would do the following:

- load the CSV,
- clean and validate columns,
- map the target label to binary values if needed,
- split data into train/test sets,
- train the same models,
- log metrics with MLflow,
- save the best model,
- serve it through the prediction API.

This project is already structured so that the MLOps pipeline can be reused with a real-world dataset once the data is provided.

## Future improvements

This project can be extended with:

- real patient data validation and schema checks,
- feature engineering and preprocessing pipelines,
- model registry integration,
- CI/CD for automated retraining and deployment,
- dashboards for model drift and performance,
- secure secrets management,
- more robust monitoring and alerting,
- better explainability and clinician-facing outputs.

## Summary

This project solves the real-world problem of turning a breast cancer prediction model into a reliable, monitored, deployable ML system.

The core idea is simple:

- build a useful model,
- track experiments,
- package it into an API,
- secure it,
- monitor it,
- prepare it for deployment.

That is the heart of MLOps.

## Important note

This project is an educational and engineering demonstration. It uses a public medical dataset for model training and should not be used for clinical decision-making without proper regulatory review, validation, and domain oversight.
