import os
import pytest
import pandas as pd

from training.training import read_data, transform, best_parameters, train_best_model, main

from prefect import task, flow


test_data_path = 'test_data.csv'


#     return df
@flow
def test_read_data():
    # Define the path to the test data file
    test_data_path = 'test_data.csv'
    
    # Call the read_data method with the test data path
    df = read_data(test_data_path)
    
    # Assert that 'OtherColumn' is still present in the DataFrame
    print(df.columns)
    assert 'Churn' in df.columns
    
    # Assert that the 'cols_to_drop' columns were dropped
    cols_to_drop = ['State', 'Area code', 'Total day charge', 'Total eve charge', 
                    'Total night charge', 'Total intl charge']
    for col in cols_to_drop:
        assert col not in df.columns

@flow
def test_transform():
    df_train = read_data(test_data_path)
    df_val = read_data(test_data_path)
    X_train, X_val, y_train, y_val, dv = transform(df_train, df_val)
    assert X_train is not None
    assert X_val is not None
    assert y_train is not None
    assert y_val is not None
    assert dv is not None

