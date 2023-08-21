import pandas as pd
from  prefect import task, flow, get_run_logger
import mlflow
import pickle
import sys
import os
from dotenv import load_dotenv
import boto3


load_dotenv()


aws_access_key_id = os.environ['AWS_ACCESS_KEY_ID']
aws_secret_access_key = os.environ['AWS_SECRET_ACCESS_KEY']


session = boto3.Session(
                        aws_access_key_id=aws_access_key_id,
                        aws_secret_access_key=aws_secret_access_key)


@task(log_prints=True, name="Read data into DataFrame", retries=3, retry_delay_seconds=2)
def read_data(filename: str) -> pd.DataFrame:
    """Read data into DataFrame"""
    logger = get_run_logger()
    logger.info("Reading data...")
    df = pd.read_csv(filename)

    cols_to_drop = ['State', 'Area code', 'Total day charge', 'Total eve charge', 
               'Total night charge', 'Total intl charge']
    

    for col in cols_to_drop:
        if col in df.columns:
            df.drop(columns = col, axis = 1, inplace=True)

    print("Successfully read data into DataFrame")

    return df

@task
def transform(df, dv):

    train_dicts = df.to_dict(orient='records')

    X = dv.transform(train_dicts)

    return X


@task
def get_model(model_uri, artifact_uri):
    print("Getting model...")
    model = mlflow.pyfunc.load_model(model_uri)
    artifact = mlflow.artifacts.download_artifacts(artifact_uri)
    with open(artifact, "rb") as file:
        dv = pickle.load(file)
    return model, dv

@task
def batch_scoring(model, df):
    y_preds = model.predict(df)
    return y_preds

@task
def save_predictions(df, pred, predictions):
    df['Churn'] = pred
    df.to_csv(predictions)
    return

@flow
def churn_prediction(dummy_data, prediction_data):

    data_path=os.environ["DATA_PATH"]
    dummy_data = f'{data_path}/data/{dummy_data}'
    predictions_path = os.environ["PREDICTIONS_PATH"]
    predictions= f'{predictions_path}/{prediction_data}'


    model_uri = os.environ["MODEL_URI"]
    preprocessor_uri = os.environ["PREPROCESSOR_URI"]

    mlflow.set_tracking_uri(os.environ["TRACKING_URI"])

    df = read_data(dummy_data)
    model, dv = get_model(model_uri, preprocessor_uri)
    X = transform(df,dv)
    pred = batch_scoring(model, X) 

    save_predictions(df, pred, predictions)
    return


def main():
    dummy_data  = sys.argv[1]
    prediction_data = sys.argv[2]
    churn_prediction(dummy_data, prediction_data)

    return


if __name__ =="__main__":
    main()