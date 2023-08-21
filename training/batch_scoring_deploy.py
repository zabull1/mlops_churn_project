from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule
from training.batch_scoring import churn_prediction

deployment = Deployment.build_from_flow(
    flow=churn_prediction,
    name="churn_prediction",
    parameters ={"dummy_data": "dummy_prod1.csv", "prediction_data": "predictions_deploy.csv"},
    schedule=CronSchedule(cron="0 3 2 * *"),
    work_queue_name="ml",
)

deployment.apply()
