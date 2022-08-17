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

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)
from sklearn.pipeline import make_pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.feature_selection import (
    SelectFromModel,
    SelectKBest,
    SequentialFeatureSelector,
    VarianceThreshold,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold

import xgboost as xgb

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

# %run notebooks/amex-feature-engineering.ipynb import load_dataset, features_dict

# +
# https://www.kaggle.com/kyakovlev
# https://www.kaggle.com/competitions/amex-default-prediction/discussion/327534
def amex_score(y_true, y_pred):
    """
    > The function takes in the true labels and the predicted labels, and returns the evaluation  score. (TODO: correct the following passage) The evaluation metric is the average of the Gini coefficient and the top 4% score

    Args:
      y_true: the true labels
      y_pred: the predicted probabilities of the positive class

    Returns:
      the score of the model.
    """
    labels = np.transpose(np.array([y_true, y_pred]))
    labels = labels[labels[:, 1].argsort()[::-1]]
    weights = np.where(labels[:, 0] == 0, 20, 1)
    cut_vals = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four = np.sum(cut_vals[:, 0]) / np.sum(labels[:, 0])

    gini = [0, 0]
    for i in [1, 0]:
        labels = np.transpose(np.array([y_true, y_pred]))
        labels = labels[labels[:, i].argsort()[::-1]]
        weight = np.where(labels[:, 0] == 0, 20, 1)
        weight_random = np.cumsum(weight / np.sum(weight))
        total_pos = np.sum(labels[:, 0] * weight)
        cum_pos_found = np.cumsum(labels[:, 0] * weight)
        lorentz = cum_pos_found / total_pos
        gini[i] = np.sum((lorentz - weight_random) * weight)

    # Returning the average of the Gini coefficient and the top 4% score.
    return 0.5 * (gini[1] / gini[0] + top_four)


def xgb__amex_metric(labels, predt):
    """
    It takes in the actual values and the predicted values and returns the score

    Args:
      labels: the actual values of the target variable
      predt: the predictions from the model

    Returns:
      the score of the model.
    """
    score = 1 - amex_score(labels, predt)
    return score


# -


def make_preprocessor(X_train):
    """
    It takes the training dataset as input and returns a pipeline that can be used to
    preprocess the training data and the test data.

    Args:
      X_train: The training data

    Returns:
      A pipeline object
    """
    # Creating a dictionary of the features in the training set.
    features = features_dict(X_train)

    # Impute missing values in numeric features with their median values and then adding an indicator column to indicate which values were imputed.
    numeric_preprocessor = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True)
    )

    # Impute missing values in categorical features with their most frequent value before one-hot-encoding the categorical features.
    categorical_preprocessor = make_pipeline(
        SimpleImputer(strategy="most_frequent", add_indicator=True),
        OneHotEncoder(handle_unknown="ignore", sparse=False),
    )

    # Impute missing values in ordinal features with -1 and then encoding the ordinal features.
    ordinal_preprocessor = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=-1, add_indicator=True),
        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-11),
    )

    # Make ColumnTransformer to combine various numeric/categorical transformers
    feature_preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_preprocessor, features["numeric"]),
            ("categorical", categorical_preprocessor, features["categorical"]),
        ],
        verbose_feature_names_out=True,
    )

    # Select the top 750 features from the training set using the Random Forest Classifier.
    feature_selector = SelectFromModel(
        RandomForestClassifier(n_estimators=25, random_state=1123), max_features=750
    )

    # Make the final pipeline to preprocess the training and test datasets.
    preprocessor_pipeline = make_pipeline(feature_preprocessor)

    # Return the pipeline object.
    return preprocessor_pipeline


# +
# Print version of XGBoost used
print(f"XGB Version: {xgb.__version__}")

# Instantiate the XGBClassifier
xgb_clf = xgb.XGBClassifier(
    objective="binary:logistic",
    booster="dart",
    use_label_encoder=False,
    max_depth=7,
    early_stopping_rounds=5,
    subsample=0.88,
    colsample_bytree=0.72,
    n_estimators=128,
    learning_rate=0.32,
    feval=amex_score,
    eval_metric=xgb__amex_metric,
    verbosity=3,
    seed=1123,
    n_jobs=-1,
)
# -

# Load training data from the `data/processed` directory.
amex_train__agg = load_dataset("train_agg", use_feather=True)

# +
# Split the data into train and test sets.
X_train, X_test, y_train, y_test = train_test_split(
    amex_train__agg.drop("target", axis=1),
    amex_train__agg.target,
    stratify=amex_train__agg.target,
    test_size=0.20,
    random_state=1123,
)

# Delete the `amex_train__agg` variable and then call the garbage collector to free up memory.
del amex_train__agg
gc.collect()

# +
# Make a pipeline object that can be used to preprocess the training and test datasets.
feature_preprocessor = make_preprocessor(X_train)

# Fit the preprocessor to the training data.
feature_preprocessor.fit(X_train, y_train)

# Preprocess the training and test datasets.
X_train__preprocessed = feature_preprocessor.transform(X_train)
X_test__preprocessed = feature_preprocessor.transform(X_test)
# -

# Fit the classifier to the training set
xgb_clf.fit(X_train__preprocessed, y_train, eval_set=[(X_test__preprocessed, y_test)])

# +
# Predict the probabilities for the positive class in the training and test datasets.
train_preds = xgb_clf.predict_proba(X_train__preprocessed)[:, 1]
test_preds = xgb_clf.predict_proba(X_test__preprocessed)[:, 1]

# Calculate the evaluation metric for the training and test datasets.
train_score = amex_score(y_train.values, train_preds)
test_score = amex_score(y_test.values, test_preds)

# Print model scores for both the training and test datasets.
print(f"Train Score: {train_score}")
print(f"Test Score: {test_score}")

# +
# Save the `feature_preprocessor` object to a file called `feature_preprocessor.pkl` in the `models` directory.
with open("models/feature_preprocessor.pkl", "wb") as f:
    pickle.dump(feature_preprocessor, f)

# Save the `xgb_clf` object to a file called `xgb_clf.pkl` in the `models` directory.
with open("models/xgb_clf.pkl", "wb") as f:
    pickle.dump(xgb_clf, f)
