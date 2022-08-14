# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3.9.12 ('base')
#     language: python
#     name: python3
# ---

# +
# Import required packages
import os
import gc
import pickle

import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# +
# Change working directory to project root
if os.getcwd().split("/")[-1] == "notebooks":
    os.chdir("../")
    
# Print the current working directory
print(f'Current Dir: {os.getcwd()}')
    
# Enable garbage collection
gc.enable()

# Configure display options for Pandas
# (*) Helpful when displaying DFs w/ numerous features
pd.set_option("display.width", 1000)
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)


# -

def load_dataset(dataset, use_feather=True):
    # Function to load amex data
    # Raw Data - Source: American Express via Kaggle
    # Reduced  - Source: https://www.kaggle.com/datasets/munumbutt/amexfeather
    valid_datasets = ("train", "test", "train_agg", "test_agg")
    if not isinstance(dataset, str):
        raise TypeError
    elif dataset not in valid_datasets:
        raise ValueError
    fpaths = {
        "feather": {
            "train": "./data/external/train_data.ftr",
            "test": "./data/external/test_data.ftr",
            "train_agg": "./data/interim/train_agg.ftr",
            "test_agg": "./data/interim/test_agg.ftr",
        },
        "csv": {
            "train": "./data/raw/train_data.csv",
            "test": "./data/raw/test_data.csv",
            "train_agg": "./data/interim/train_agg.csv",
            "test_agg": "./data/interim/test_agg.csv",
        },
    }
    if use_feather:
        feather_file = fpaths["feather"].get(dataset)
        return pd.read_feather(feather_file).set_index("customer_ID")
    else:
        csv_file = fpaths["csv"].get(dataset)
        return pd.read_csv(csv_file, index_col="customer_ID")


def get_incomplete_features(amex_dataset, threshold=0.85, verbose=True):
    if not isinstance(threshold, (float, int)):
        raise TypeError()
    elif (threshold < 0) or (threshold > 1):
        raise ValueError()
    pct_incomplete = amex.isnull().sum().div(len(amex)).sort_values(ascending=False)
    incomplete_features = set(
        pct_incomplete[pct_incomplete >= threshold].index.tolist()
    )
    if verbose:
        print(f"Incomplete Features >= {threshold}%:\n{incomplete_features}")


def create_agg_features(amex_dataset):
    amex_numeric = amex_dataset.select_dtypes(include="number")

    last_statement = amex_numeric.groupby("customer_ID").nth(-1)
    first_statement = amex_numeric.groupby("customer_ID").nth(0)

    lag_div = last_statement.div(first_statement).fillna(1)
    lag_diff = last_statement.subtract(first_statement).fillna(0)

    lag_div.columns = [col + "__lag_div" for col in lag_div.columns]
    lag_diff.columns = [col + "__lag_diff" for col in lag_diff.columns]

    numeric__agg_lag = pd.concat([lag_diff, lag_div], axis=1)

    numeric__agg = amex_numeric.groupby("customer_ID").agg(
        ["first", "last", "mean", "min", "max", "std", "sem"]
    )
    numeric__agg.columns = ["__".join(col) for col in numeric__agg.columns]

    categorical__agg = (
        amex_dataset.select_dtypes(include="category")
        .groupby("customer_ID")
        .agg(["first", "last", "count", "nunique"])
    )
    categorical__agg.columns = ["__".join(col) for col in categorical__agg.columns]

    amex_agg = pd.concat([categorical__agg, numeric__agg, numeric__agg_lag], axis=1)

    return amex_agg

