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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.feature_selection import SelectFromModel, SelectKBest, SequentialFeatureSelector, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold

import xgboost as xgb

# +
# Change working directory to project root
if os.getcwd().split("/")[-1] == "notebooks":
    # Change working dir to project root
    os.chdir("../")
    
    # Print the current working directory
    # print(f'Current Dir: {os.getcwd()}')
    
# Enable garbage collection
gc.enable()

# Configure display options for Pandas
# (*) Helpful when displaying DFs w/ numerous features
pd.set_option("display.width", 1000)
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)


# -

# %run notebooks/amex-feature-engineering.ipynb import load_dataset, features_dict

# +
# https://www.kaggle.com/kyakovlev
# https://www.kaggle.com/competitions/amex-default-prediction/discussion/327534
def amex_score(y_true, y_pred):

    labels     = np.transpose(np.array([y_true, y_pred]))
    labels     = labels[labels[:, 1].argsort()[::-1]]
    weights    = np.where(labels[:,0]==0, 20, 1)
    cut_vals   = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four   = np.sum(cut_vals[:,0]) / np.sum(labels[:,0])

    gini = [0,0]
    for i in [1,0]:
        labels         = np.transpose(np.array([y_true, y_pred]))
        labels         = labels[labels[:, i].argsort()[::-1]]
        weight         = np.where(labels[:,0]==0, 20, 1)
        weight_random  = np.cumsum(weight / np.sum(weight))
        total_pos      = np.sum(labels[:, 0] *  weight)
        cum_pos_found  = np.cumsum(labels[:, 0] * weight)
        lorentz        = cum_pos_found / total_pos
        gini[i]        = np.sum((lorentz - weight_random) * weight)

    return 0.5 * (gini[1]/gini[0] + top_four)


def xgb__amex_metric(labels, predt):
    score = 1 - amex_score(labels, predt)
    return score


# -

def make_preprocessor(X_train):
    features = features_dict(X_train)

    numeric_preprocessor = make_pipeline(
        SimpleImputer(strategy='median', add_indicator=True))

    categorical_preprocessor = make_pipeline(
        SimpleImputer(strategy='most_frequent', add_indicator=True),
        OneHotEncoder(handle_unknown='ignore', sparse=False))

    ordinal_preprocessor = make_pipeline(
        SimpleImputer(strategy='constant', fill_value=-1, add_indicator=True),
        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-11))

    feature_preprocessor = ColumnTransformer([
            ('numeric', numeric_preprocessor, features['numeric']),
            ('categorical', categorical_preprocessor, features['categorical'])
        ], verbose_feature_names_out=True)

    feature_selector = SelectFromModel(
        RandomForestClassifier(
            n_estimators=25,
            random_state=1123), 
        max_features=750)

    preprocessor_pipeline = make_pipeline(
        feature_preprocessor)
    
    return preprocessor_pipeline


# +
# Print version of XGBoost used
print(f'XGB Version: {xgb.__version__}')

# Instantiate the XGBClassifier
xgb_clf = xgb.XGBClassifier(
    objective='binary:logistic',
    booster='dart',
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
    seed=1123, n_jobs=-1)
# -

amex_train__agg = load_dataset('train_agg', use_feather=True)

# +
X_train, X_test, y_train, y_test = train_test_split(
    amex_train__agg.drop('target', axis=1), 
    amex_train__agg.target,
    stratify=amex_train__agg.target,
    test_size=0.20,
    random_state=1123)

del amex_train__agg
gc.collect()

# +
feature_preprocessor = make_preprocessor(X_train)

feature_preprocessor.fit(X_train, y_train)

X_train__preprocessed = feature_preprocessor.transform(X_train)
X_test__preprocessed = feature_preprocessor.transform(X_test)
# -

# Fit the classifier to the training set
xgb_clf.fit(X_train__preprocessed, y_train, 
            eval_set=[(X_test__preprocessed, y_test)])

# +
# Predict the labels of the test set: preds
train_preds = xgb_clf.predict_proba(X_train__preprocessed)[:,1]
test_preds = xgb_clf.predict_proba(X_test__preprocessed)[:,1]

train_score = amex_score(y_train.values, train_preds)
test_score = amex_score(y_test.values, test_preds)

print(f'Train Score: {train_score}')
print(f'Test Score: {test_score}')

# +
with open('models/feature_preprocessor.pkl','wb') as f:
    pickle.dump(feature_preprocessor, f)

with open('models/xgb_clf.pkl','wb') as f:
    pickle.dump(xgb_clf, f)
