# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
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
    # Change dir to parent directory.
    os.chdir("../")
    # Print the current working directory
    print(f'Current Dir: {os.getcwd()}')
    
# Enable garbage collection
gc.enable()

# > Configure display options for Pandas
# ---------------------------------------------------------------
# Set display width to 1000.
pd.set_option("display.width", 1000)
# Set maximum number of rows to display to 500.
pd.set_option("display.max_rows", 500)
# Set maximum number of columns to display to 500.
pd.set_option("display.max_columns", 500)


# -

def load_dataset(dataset, use_feather=True):
    '''> Function takes in the name of a dataset, and then returns the feather or csv file for the dataset. [TODO: update content]
    
    Parameters
    ----------
    dataset
        The dataset to load.
    use_feather, optional
        A boolean that determines whether to use the feather file or the csv file.
    
    Returns
    -------
        the DataFrame for the dataset.
    
    '''
    # Function to load amex data
    # Raw Data - Source: American Express via Kaggle
    # Reduced  - Source: https://www.kaggle.com/datasets/munumbutt/amexfeather

    # Tuple of strings that are valid dataset names that can be loaded.
    valid_datasets = ("train", "test", "train_agg", "test_agg")

    # Check `dataset` parameter type. If the type is not a string, then raise a TypeError. 
    # If dataset is not in the list of valid datasets, then raise a ValueError.
    if not isinstance(dataset, str):
        raise TypeError
    elif dataset not in valid_datasets:
        raise ValueError

    # A dictionary of featureset dictionaries. 
    # The outer dictionary has two keys to select data format: 
    # ----> `feather`, `csv`
    # The inner dictionaries have four keys corresponding to datasets: 
    # ----> `train`, `test`, `train_agg`, `test_agg`
    # Values correspond to file paths for feather and csv files for that dataset.
    data_filepaths = {
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
        # Set path to feather file for the dataset.
        feather_file = data_filepaths["feather"].get(dataset)
        # Return feather file as Pandas DataFrame; set index to the customer ID.
        return pd.read_feather(feather_file).set_index("customer_ID")
    else:
        # Set path to *.csv file for the dataset.
        csv_file = data_filepaths["csv"].get(dataset)
        # Return csv file as Pandas DataFrame using customer_ID as the index column
        return pd.read_csv(csv_file, index_col="customer_ID")


def get_incomplete_features(amex_dataset, threshold=0.85, verbose=True):
    '''> Function takes in a dataset and a threshold value, and returns a set of features that have a percentage of missing values that is greater than or equal to the threshold. [TODO: update content]
    
    Parameters
    ----------
    amex_dataset
        The dataset that we want to check for incomplete features.
    threshold
        The percentage of missing values that a feature must have to be considered incomplete.
    verbose, optional
        If True, prints the number of features in each category.
    
    '''
    # Check the type and that the value of the `threshold`` parameter is in range: 
    if not isinstance(threshold, (float, int)):
        # If invalid (not a float or an integer), than raise a TypeError. 
        raise TypeError()
    elif (threshold < 0) or (threshold > 1):
        # If invalid (not between 0 and 1), than raise a ValueError.
        raise ValueError()

    # Create a Series containing the percentage missing or null for each column in the dataset.
    pct_incomplete = amex.isnull().sum().div(len(amex)).sort_values(ascending=False)

    # Create set containing incomplete features, or column names where the percentage of missing or null values are greater than or equal to the assigned threshold.
    incomplete_features = set(
        pct_incomplete[pct_incomplete >= threshold].index.tolist()
    )
    # Print the number of incomplete features for a given threshold.
    if verbose:
        print(f"Incomplete Features >= {threshold}%:\n{incomplete_features}")


def create_agg_features(amex_dataset):
    '''Function to create new features by aggregating the dataset by [TODO: add content].
    
    Parameters
    ----------
    amex_dataset
        The dataset that we want to create the aggregated features for.
    
    Returns
    -------
        DataFrame of aggregated features derived from the original dataset.
    
    '''
    # Select numeric features from the dataset
    amex_numeric = amex_dataset.select_dtypes(include="number")


    # Creating a new dataframe with the first and last rows of the dataframe.
    last_statement = amex_numeric.groupby("customer_ID").nth(-1)
    first_statement = amex_numeric.groupby("customer_ID").nth(0)

    # Dividing the last statement by the first statement and filling the NaN values with 1.
    lag_div = (last_statement
               .div(first_statement)
               .replace(-np.inf, 0)
               .replace(np.inf, 2)
               .fillna(1))

    # Subtracting the first statement from the last statement and filling the NaN values with 0.
    lag_diff = (last_statement
                .subtract(first_statement)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0))

    # Creating a new column name for each column in the `lag_div` dataframe.
    lag_div.columns = [col + "__lag_div" for col in lag_div.columns]
    lag_diff.columns = [col + "__lag_diff" for col in lag_diff.columns]

    # Creating a new dataframe by concatenating the `lag_diff` and `lag_div` dataframes.
    numeric__agg_lag = pd.concat([lag_diff, lag_div], axis=1)

    
    numeric__agg = amex_numeric.groupby("customer_ID").agg(
        ["first", "last", "mean", "min", "max", "std", "sem"]
    )
    numeric__agg.columns = ["__".join(col) for col in numeric__agg.columns]

    # Group data by the customer ID; 
    # Select the columns with categorical data; 
    # Lastly, apply aggregate functions.
    categorical__agg = (
        amex_dataset.select_dtypes(include="category")
        .groupby("customer_ID")
        .agg(["first", "last", "count", "nunique"])
    )

    # Creating a new column name for each column in the `categorical__agg` dataframe.
    categorical__agg.columns = ["__".join(col) for col in categorical__agg.columns]

    # Concatenating the `categorical__agg`, `numeric__agg`, and `numeric__agg_lag` dataframes.
    amex_agg = pd.concat([categorical__agg, numeric__agg, numeric__agg_lag], axis=1)

    # Returning the aggregated features for the dataset.
    return amex_agg


def make_features(amex_dataset):
    '''> Function takes in the amex dataset as a DataFrame; drops columns that are invalid, creates aggregated features for the dataset, and returns a DataFrame with the aggregated features.
    
    Parameters
    ----------
    amex_dataset
        The dataset that we want to create the aggregated features for.
    
    Returns
    -------
        The aggregated features for the dataset.
    
    '''
    # Create a set consisting of incomplete features.
    incomplete_features = {
        'D_87', 'D_88', 'D_108', 'D_110', 'D_111', 'B_39', 
        'D_73', 'B_42', 'D_134', 'D_137', 'D_135', 'D_138', 
        'D_136', 'R_9', 'B_29', 'D_106', 'D_132', 'D_49', 
        'R_26', 'D_76', 'D_66', 'D_42'}

    # Create a set of features that are now redundant.
    made_redundant = {'S_2'}

    # Create a set for the target variable.
    target_variable = {'target'}

    # Create a tuple of invalid features by taking the union of the three sets
    invalid_cols = (incomplete_features | made_redundant | target_variable)

    # Create a list of columns that are in both the `amex_dataset` and `invalid_cols`
    cols_to_drop = (amex_dataset
                    .columns
                    .intersection(invalid_cols)
                    .tolist())

    # 1) Drop columns that in `cols_to_drop` from the dataset
    # 2) Create aggregated features for the dataset.
    amex_aggregated = (amex_dataset
                       .drop(cols_to_drop, axis=1)
                       .pipe(create_agg_features))
        
    if 'target' in cols_to_drop:
        # Add target variable to the DataFrame
        amex_aggregated['target'] = (
            amex_dataset.groupby('customer_ID').tail(1).target)
    # Return DataFrame w/ aggregated features
    return amex_aggregated


def make_datasets(read_feather=True, to_feather=True, to_csv=False):
    '''Function loads the train and test datasets, makes features, and saves them to feather files
    [TODO: add content]
    Parameters
    ----------
    read_feather, optional
        whether to read the feather files or not
    to_feather, optional
        whether to save the dataframe to feather format
    to_csv, optional
        If True, will save the dataframes to csv files.
    
    '''
    # Load train and test datasets in feather format; pipe to make features for modeling.
    train_agg = (load_dataset('train', use_feather=read_feather)
                 .pipe(make_features))
    test_agg = (load_dataset("test", use_feather=read_feather)
                .pipe(make_features))
    
    if to_feather:
        # Save train_agg and test_agg datasets to disk as feather files.
        train_agg.reset_index().to_feather('./data/interim/train_agg.ftr')
        test_agg.reset_index().to_feather('./data/interim/test_agg.ftr')
        
    if to_csv:
        # Save train_agg and test_agg datasets to disk as csv files.
        train_agg.to_csv('./data/interim/train_agg.csv')
        test_agg.to_csv('./data/interim/test_agg.csv')


def features_dict(X_train, verbose=True):
    '''Function takes a dataset as a DataFrame and returns a dictionary of feature dictionaries.
    [TODO: update content]
    Parameters
    ----------
    X_train
        The training dataset.
    verbose, optional
        If True, it will print the length of the numeric_features, categorical_features, and
    ordinal_features.
    
    Returns
    -------
        A dictionary with the keys and values of the numeric_features, categorical_features,
    ordinal_features, and all_features.
    
    '''
    # Select all numeric data within the dataset; access their column names; cast as list.
    numeric_features = (X_train
                        .select_dtypes(include='number')
                        .columns
                        .tolist())

    # Select all categorical data within the dataset; access their column names; cast as list.
    categorical_features = (X_train
                            .select_dtypes(include='category')
                            .columns
                            .tolist())

    # Create an empty list of ordinal features (for later use).
    ordinal_features = []
    
    # Create list of all features by assigning the column names in the dataset to a list.
    all_features = X_train.columns.tolist()
    
    if verbose:
        #Print the number of features with numeric, categorical, or ordinal data.
        print(f'Numeric Features - Count(#): {len(numeric_features)}')
        print(f'Categorical Features - Count(#): {len(categorical_features)}')
        print(f'Ordinal Features - Count(#): {len(ordinal_features)}')

    # Return the dictionary of feature dictionaries
    return {
        'num': numeric_features,
        'cat': categorical_features,
        'ord': ordinal_features,
        'all': all_features,
        'numeric': numeric_features,
        'categorical': categorical_features,
        'ordinal': ordinal_features,
        'all_features': all_features
    }
