import os
import time
from dotenv import load_dotenv

import mlflow
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier

from classes.data_processor import DataPreprocessor
from classes.model_trainer import ModelTrainer

if __name__ == "__main__":
    print("doctor cancellation prediction launched")

    # Load the environment variables
    load_dotenv()

    # Init for MLflow
    APP_URI = os.getenv("APP_URI")
    print(f"Call MLflow URI: {APP_URI}")
    EXPERIMENT_NAME = "doctor-cancellation-detector"

    mlflow.set_tracking_uri(APP_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Get our experiment info
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    # Time execution
    start_time = time.time()

    # Load data
    rawdata = pd.read_csv("./data/rawdata.zip")

    # Prepare data
    dataset = rawdata.drop(["Unnamed: 0", "PatientId", "AppointmentID"], axis=1)
    dataset["AppointmentDay"] = pd.to_datetime(dataset["AppointmentDay"])
    dataset["ScheduledDay"] = pd.to_datetime(dataset["ScheduledDay"])
    dataset["diff_appointment_scheduled"] = (
        dataset["AppointmentDay"] - dataset["ScheduledDay"]
    ).dt.days
    dataset["diff_appointment_scheduled"] = dataset["diff_appointment_scheduled"].apply(
        lambda x: 0 if x == -1 else x
    )
    dataset["AppointmentDay_DayOfWeek"] = dataset["AppointmentDay"].dt.day_of_week
    dataset["AppointmentDay_Month"] = dataset["AppointmentDay"].dt.month
    dataset = dataset.drop(columns=["AppointmentDay", "ScheduledDay"])
    dataset["No-show"] = dataset["No-show"].apply(lambda x: 0 if x == "No" else 1)

    # Create the model
    model = RandomForestClassifier()
    # Set model params
    n_estimators = 100
    max_depth = 64
    min_samples_split = 4
    random_state = 42
    n_jobs = -1

    model.set_params(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    # Enable mlflow autolog
    mlflow.sklearn.autolog()

    # Log experiment to MLFlow
    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
        # Preprocess data
        preprocessor = DataPreprocessor(df=dataset, target_column="No-show")
        preprocessor.preprocess()

        model_name = type(model).__name__

        # Log Params
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("n_jobs", n_jobs)

        # training model
        trainer = ModelTrainer(preprocessor, model)
        accuracy_score, f1_score = trainer.train()

        # Log metrics
        mlflow.log_metric("test_accuracy_score", accuracy_score)
        mlflow.log_metric("test_f1_score", f1_score)

        # Log other model performance metrics or custom information
        mlflow.log_param("test_size", trainer.test_size)

        predictions = trainer.model.predict(trainer.X_train.toarray()[:1])

        print(predictions)

        # Log the sklearn model and register as version
        mlflow.sklearn.log_model(
            sk_model=trainer.model,  # Note: model should be the trained instance
            artifact_path=EXPERIMENT_NAME,
            registered_model_name=model_name,  # Choose a meaningful name
            signature=infer_signature(
                preprocessor.X[:1],
                predictions,
            ),
        )

print("...Done!")
