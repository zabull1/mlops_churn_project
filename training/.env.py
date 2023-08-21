# AWS Credentials
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=

# AWS Configuration
AWS_REGION=us-east-1


# MLflow Configuration
TRACKING_SERVER_HOST=127.0.0.1 # fill in with the public DNS of the EC2 instance
TRACKING_SERVER_PORT=5000
TRACKING_URI=http://${TRACKING_SERVER_HOST}:${TRACKING_SERVER_PORT}
EXPERIMENT_NAME=mlops_churn_prediction


# Model Configuration
MODEL_NAME=churn_prediction_model
RUN_ID=17a9920fc9fb43c6b3cc590b6b7b9447

# S3 Configurations
S3_BUCKET=s3://churn-model-bucket-mlops-zoomcamp
DATA_PATH=s3://data-churn

# Data Paths
TRAINING_DATASET=${DATA_PATH}/data/churn-bigml-80.csv.xls
VALIDATION_DATASET=${DATA_PATH}/data/churn-bigml-20.csv.xls
PREDICTIONS_PATH=${DATA_PATH}/predictions/${RUN_ID}



# Model and artifact URIs
MODEL_URI=${S3_BUCKET}/1/${RUN_ID}/artifacts/model
PREPROCESSOR_URI=${S3_BUCKET}/1/${RUN_ID}/artifacts/preprocessor/preprocessor.b