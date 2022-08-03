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
#     display_name: Python 3.9.12 ('base')
#     language: python
#     name: python3
# ---

# # American Express - Default Prediction
# Whether out at a restaurant or buying tickets to a concert, modern life counts on the convenience of a credit card to make daily purchases. It saves us from carrying large amounts of cash and also can advance a full purchase that can be paid over time. How do card issuers know we’ll pay back what we charge? That’s a complex problem with many existing solutions—and even more potential improvements, to be explored in this competition.
# ## Introduction
# Credit default prediction is central to managing risk in a consumer lending business. Credit default prediction allows lenders to optimize lending decisions, which leads to a better customer experience and sound business economics. Current models exist to help manage risk. But it's possible to create better models that can outperform those currently in use.
# ### Objective
# In this competition, we’ll apply supervised machine learning to predict credit default. Specifically, we will leverage an industrial scale dataset to build binary classifaction models that challenge the current model in production. Training, validation, and testing datasets include: time-series, behavioral data, and anonymized customer profile information. Apart from creating a base model, we will explore numerous techniques and methodolgies to create an impressive model through feature engineering and using the data in a more organic way within a model.
# ### Evaluation Criteria
# If successful, our solution when implemented may yield better customer experiences for cardholders by making it easier for them to be approved for a new credit card. Top solutions may even challenge the credit default prediction model used by the world's largest payment card issuer at American Express.

# ### Project Setup and Configuration

# #### Notebook Configuration

# +
# Change working directory to project root
import os

if os.getcwd().split("/")[-1] == "notebooks":
    os.chdir("../")
# -

# #### Import Packages

# +
# Import required packages
import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
# -

# ### Load AMEX Datasets
# **Source (Raw Data):**
# - https://www.kaggle.com/competitions/amex-default-prediction/data
#
# **Source (Compressed Data):**
# - https://www.kaggle.com/datasets/munumbutt/amexfeather
#
# // TODO: Add information regarding the source, advantages and limitations in using the compressed datasets

# +
# Load compressed datasets
# Source: https://www.kaggle.com/datasets/munumbutt/amexfeather

train = pd.read_feather("./data/raw/train_data.ftr")
test = pd.read_feather("./data/raw/test_data.ftr")
# -

# Preview the first five rows of the training data
train.head()

# Print the shape of the DataFrame for the training set
print(f"Training Data: Shape == {train.shape}")
print(
    f"\nThe training set consists of {train.shape[0]} observations with {train.shape[1]} features and 1 target variable."
)

train.info()

# Preview the first five rows of the testing set
test.head(5)

# Print the shape of the DataFrame for the testing dataset
print(f"Testing Data: Shape == {test.shape}")
print(
    f"\nThe testing set consists of {test.shape[0]} observations with {test.shape[1]} features."
)

test.info()

# +
# 1-2) Sum the number of incomplete (missing or null) values in each column
# 3-4) Divide by the number of observations and multipy by 100 to make it a percentage.
#   5) Lastly, sort the values in descending order to better observe feature incompleteness.
pct_incomplete = (
    train.isna().sum().div(len(train)).mul(100).sort_values(ascending=False)
)

# Subset pct_incomplete to select incomplete features (Threshold: >20%)
incomplete_features = set(pct_incomplete[pct_incomplete >= 20].index)

# +
# Print the count of incomplete features
print(
    f"{len(incomplete_features)} features with over 20% values are missing or null.\n"
)

# Print column names of features where 20% or greater have missing or null values
print(f"Incomplete Features: \n{incomplete_features}")
# -


