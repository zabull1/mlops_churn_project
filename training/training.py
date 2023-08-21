#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import (precision_score, 
                        recall_score,
                        f1_score,
                        roc_auc_score, 
                        average_precision_score)
import mlflow
from hyperopt import fmin, hp, tpe, STATUS_OK, Trials
import xgboost as xgb
from prefect import flow, get_run_logger, task
import boto3
from dotenv import load_dotenv
import matplotlib
matplotlib.use('agg')




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
def transform(df_train, df_val):
    """Transform DataFrame"""
    dv = DictVectorizer()

    train_dicts = df_train.to_dict(orient='records')

    val_dicts = df_val.to_dict(orient='records')

    X_train = dv.fit_transform(train_dicts)

    X_val = dv.fit_transform(val_dicts)

    y_train = df_train['Churn'].values

    y_val = df_val['Churn'].values

    return X_train, X_val, y_train, y_val, dv

@task
def best_parameters(X_train, y_train):
    """Search for best parameters"""
    search_space = {
    "objective": "binary:logistic",
    "max_depth": hp.choice("max_depth", np.arange(1, 100, dtype=int)),
    "min_child_weight": hp.uniform("min_child_weight", 0, 5),
    "learning_rate": hp.loguniform("learning_rate", np.log(0.005), np.log(0.2)),
    "gamma": hp.uniform("gamma", 0, 5),
    "colsample_bytree": hp.quniform("colsample_bytree", 0.1, 1, 0.01),
    "colsample_bynode": hp.quniform("colsample_bynode", 0.1, 1, 0.01),
    "colsample_bylevel": hp.quniform("colsample_bylevel", 0.1, 1, 0.01),
    "subsample": hp.quniform("subsample", 0.5, 1, 0.05),
    "reg_alpha": hp.uniform("reg_alpha", 0, 5),
    "reg_lambda": hp.uniform("reg_lambda", 0, 5),

    }

    def xgboost_objective_function(params):

        train = xgb.DMatrix(X_train, label=y_train)

        res = xgb.cv(
            params,
            train,
            num_boost_round=100,
            nfold=5,
            metrics={"auc"},
            seed=42,
       
        )
        best_loss = res["test-auc-mean"].iloc[-1]
        return {"loss": best_loss, "status": STATUS_OK}
    
    best_result = fmin(
    fn=xgboost_objective_function,
    space=search_space,
    algo=tpe.suggest,
    max_evals=50,
    trials=Trials()
    ) 

    print(best_result)

    return best_result 


    

@task
def train_best_model(X_train, X_val, y_train, y_val, dv, best_params):
    """train the best model"""
    with mlflow.start_run():

        mlflow.xgboost.autolog()

        mlflow.log_params(best_params)

        booster = xgb.XGBClassifier(**best_params)

        booster.fit(X_train, y_train)

        y_pred = booster.predict(X_val)
        y_pred_binary = (y_pred > 0.5).astype(int)  # Convert to binary predictions

        # Calculate classification metrics
        precision = precision_score(y_val, y_pred_binary)
        recall = recall_score(y_val, y_pred_binary)
        f1 = f1_score(y_val, y_pred_binary)
        roc_auc = roc_auc_score(y_val, y_pred)
        pr_auc = average_precision_score(y_val, y_pred)

        # Log metrics
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("pr_auc", pr_auc)

        with open("preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("preprocessor.b", artifact_path="preprocessor")
      
    return

@flow
def main():
    """ main method """
    # Load environment variables from .env file
    load_dotenv()

    aws_access_key_id = os.environ['AWS_ACCESS_KEY_ID']
    aws_secret_access_key = os.environ['AWS_SECRET_ACCESS_KEY']

    boto3.Session(
                            aws_access_key_id=aws_access_key_id,
                            aws_secret_access_key=aws_secret_access_key)

    train_path  = os.environ['TRAINING_DATASET']
    val_path = os.environ['VALIDATION_DATASET']

  
    TRACKING_URI =  os.environ["TRACKING_URI"]
    EXPERIMENT_NAME = os.environ["EXPERIMENT_NAME"]

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df_train = read_data(train_path)
    df_val = read_data(val_path)

    # Transform
    X_train, X_val, y_train, y_val, dv = transform(df_train, df_val)

    best_params = best_parameters(X_train,y_train)


    # Train
    train_best_model(X_train, X_val, y_train, y_val, dv, best_params)



if __name__ == "__main__":
    main()



