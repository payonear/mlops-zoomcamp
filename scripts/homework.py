import argparse
import os
import pickle
from datetime import datetime

import pandas as pd
from dateutil.relativedelta import relativedelta
from prefect import flow, get_run_logger, task
from prefect.deployments import DeploymentSpec
from prefect.flow_runners import SubprocessFlowRunner
from prefect.orion.schemas.schedules import CronSchedule
from prefect.task_runners import SequentialTaskRunner
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


@task
def read_data(path):
    df = pd.read_parquet(path)
    return df


@task
def prepare_features(df, categorical, train=True):
    df["duration"] = df.dropOff_datetime - df.pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        print(f"The mean duration of training is {mean_duration}")
    else:
        print(f"The mean duration of validation is {mean_duration}")

    df[categorical] = df[categorical].fillna(-1).astype("int").astype("str")
    return df


@task
def train_model(df, categorical):
    logger = get_run_logger()

    train_dicts = df[categorical].to_dict(orient="records")
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv


@task
def run_model(df, categorical, dv, lr):
    logger = get_run_logger()

    val_dicts = df[categorical].to_dict(orient="records")
    X_val = dv.transform(val_dicts)
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return


@task
def get_paths(date=None):
    logger = get_run_logger()
    if date is None:
        date = datetime.now()
    else:
        date = datetime.strptime(date, "%Y-%m-%d").date()
    logger.info(f"Parameter date value is {date.strftime('%Y-%m-%d')}")
    logger.info(f"Current dir: {os.listdir('./data')}")
    train_date = (date - relativedelta(months=2)).strftime("%Y-%m")
    val_date = (date - relativedelta(months=1)).strftime("%Y-%m")
    train_path = f"./data/fhv_tripdata_{train_date}.parquet"
    val_path = f"./data/fhv_tripdata_{val_date}.parquet"
    return train_path, val_path


@flow(name="Homework 3 flow", task_runner=SequentialTaskRunner())
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None)
    args = parser.parse_args()
    train_path, val_path = get_paths(args.date).result()

    categorical = ["PUlocationID", "DOlocationID"]

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()

    os.makedirs("./model_artifacts", exist_ok=True)

    with open(f"./model_artifacts/model-{args.date}.bin", "wb") as f_out:
        pickle.dump(lr, f_out)

    with open(f"./model_artifacts/dv-{args.date}.b", "wb") as f_out:
        pickle.dump(dv, f_out)

    run_model(df_val_processed, categorical, dv, lr)


DeploymentSpec(
    flow=main,
    name="model_training",
    schedule=CronSchedule(cron="0 9 15 * *", timezone="Europe/Warsaw"),
    flow_runner=SubprocessFlowRunner(),
    tags=["ml"],
)
