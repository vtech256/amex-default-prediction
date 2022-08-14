# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3.10.5 ('amex_v2')
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
            "train": "./data/external/compressed/train_data.ftr",
            "test": "./data/external/compressed/test_data.ftr",
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


def make_features(amex_dataset):
    incomplete_features = {
        'D_87', 'D_88', 'D_108', 'D_110', 'D_111', 'B_39', 
        'D_73', 'B_42', 'D_134', 'D_137', 'D_135', 'D_138', 
        'D_136', 'R_9', 'B_29', 'D_106', 'D_132', 'D_49', 
        'R_26', 'D_76', 'D_66', 'D_42'}
    made_redundant = {'S_2'}
    target_variable = {'target'}

    invalid_cols = (incomplete_features | made_redundant | target_variable)
    cols_to_drop = (amex_dataset
                    .columns
                    .intersection(invalid_cols)
                    .tolist())

    amex_aggregated = (amex_dataset
                       .drop(cols_to_drop, axis=1)
                       .pipe(create_agg_features))
        
    if 'target' in cols_to_drop:
        amex_aggregated['target'] = (
            amex_dataset.groupby('customer_ID').tail(1).target)
    return amex_aggregated


def make_datasets(read_feather=True, to_feather=True, to_csv=False):
    train_agg = (load_dataset('train', use_feather=read_feather)
                 .pipe(make_features))
    test_agg = (load_dataset("test", use_feather=read_feather)
                .pipe(make_features))
    
    if to_feather:
        train_agg.reset_index().to_feather('./data/interim/train_agg.ftr')
        test_agg.reset_index().to_feather('./data/interim/test_agg.ftr')
        
    if to_csv:
        train_agg.to_csv('./data/interim/train_agg.csv')
        test_agg.to_csv('./data/interim/test_agg.csv')
