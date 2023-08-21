from evidently.report import Report
from evidently.report import Report
from evidently import ColumnMapping
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric
import psycopg
from prefect import flow, get_run_logger, task
import pandas as pd
import datetime
import time
from dotenv import load_dotenv
import os
# sys.path.append('../training')
# import training 

load_dotenv()

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

SEND_TIMEOUT = 10
create_table_statement = """
DROP TABLE IF EXISTS CHURN_METRICS;
create table CHURN_METRICS(
	timestamp timestamp,
	prediction_drift float,
	num_drifted_columns integer,
	share_missing_values float
);
"""
baseline_data = os.environ['TRAINING_DATASET']

PREDICTIONS_DATA_PATH = os.environ["PREDICTIONS_PATH"]

reference_data = pd.read_csv(baseline_data).drop(columns = ['State', 'Area code', 'Total day charge', 'Total eve charge', 
               'Total night charge', 'Total intl charge'], axis =1)
raw_data = pd.read_csv(f'{PREDICTIONS_DATA_PATH}/prediction_dummy_prod.csv').drop(columns= [ "Unnamed: 0.1",  "Unnamed: 0"], axis=1)

raw_data.Churn = raw_data.Churn.apply(lambda c: c == 1)

print(raw_data.head(2))
print(reference_data.head(2))

# numerical_features = ['Account length', 'International plan', 'Voice mail plan',
#        'Number vmail messages', 'Total day minutes', 'Total day calls',
#        'Total eve minutes', 'Total eve calls', 'Total night minutes',
#        'Total night calls', 'Total intl minutes', 'Total intl calls',
#        'Customer service calls']

numerical_features = raw_data.drop(columns='Churn', axis=1).columns.to_list()

report = Report(
    metrics=[
        ColumnDriftMetric(column_name="Churn"),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
    ]
)


column_mapping = ColumnMapping(
    target=None,
    prediction='Churn',
    numerical_features= numerical_features,
    categorical_features=[]
)




@task
def prep_db():

    with psycopg.connect("host=localhost port=5432 user=postgres password=example", autocommit=True) as conn:
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname='churn_metrics'")
        if len(res.fetchall()) == 0:
            conn.execute("create database churn_metrics;")
        with psycopg.connect("host=localhost port=5432 dbname=churn_metrics user=postgres password=example") as conn:
            conn.execute(create_table_statement)

@task
def calculate_metrics_postgresql(curr, i):


	report.run(reference_data = reference_data, current_data = raw_data,
		column_mapping=column_mapping)

	result = report.as_dict()

	prediction_drift = result['metrics'][0]['result']['drift_score']
	num_drifted_columns = result['metrics'][1]['result']['number_of_drifted_columns']
	share_missing_values = result['metrics'][2]['result']['current']['share_of_missing_values']

	curr.execute(
		"insert into CHURN_METRICS(timestamp, prediction_drift, num_drifted_columns, share_missing_values) values (%s, %s, %s, %s)",
		(datetime.datetime.now(), prediction_drift, num_drifted_columns, share_missing_values)
	)

@flow
def batch_monitoring_backfill():
    logger = get_run_logger()
    prep_db()
    last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)
    with psycopg.connect("host=localhost port=5432 dbname=churn_metrics user=postgres password=example", autocommit=True) as conn:
        for i in range(0, 27):
            with conn.cursor() as curr:
                calculate_metrics_postgresql(curr, i)

            new_send = datetime.datetime.now()
            seconds_elapsed = (new_send - last_send).total_seconds()
            if seconds_elapsed < SEND_TIMEOUT:
                time.sleep(SEND_TIMEOUT - seconds_elapsed)
            while last_send < new_send:
                last_send = last_send + datetime.timedelta(seconds=10)
            logger.info("Sending metrics to database...")

if __name__ == '__main__':
	batch_monitoring_backfill()