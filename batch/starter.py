import argparse
import os
import pickle
from ast import arg

import pandas as pd

categorical = ["PUlocationID", "DOlocationID"]


def read_data(filename):
    df = pd.read_parquet(filename)

    df["duration"] = df.dropOff_datetime - df.pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype("int").astype("str")

    return df


def run(month, year):
    df = read_data(
        f"https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year:04d}-{month:02d}.parquet"
    )

    with open("model.bin", "rb") as f_in:
        dv, lr = pickle.load(f_in)

    dicts = df[categorical].to_dict(orient="records")
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)
    print(f"Mean predicted duration is {y_pred.mean()}")
    df = pd.DataFrame({"preds": y_pred})
    df["ride_id"] = f"{year:04d}/{month:02d}_" + df.index.astype("str")
    os.makedirs("output", exist_ok=True)
    output_file = "./output/duration_preds.paquet"
    df.to_parquet(output_file, engine="pyarrow", compression=None, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int)

    parser.add_argument("--month", type=int)

    args = parser.parse_args()
    run(args.month, args.year)
