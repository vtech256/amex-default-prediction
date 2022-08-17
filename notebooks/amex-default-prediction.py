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

# # American Express - Default Prediction
# Whether out at a restaurant or buying tickets to a concert, modern life counts on the convenience of a credit card to make daily purchases. It saves us from carrying large amounts of cash and also can advance a full purchase that can be paid over time. How do card issuers know we’ll pay back what we charge? That’s a complex problem with many existing solutions—and even more potential improvements, to be explored in this competition.
# ## Introduction
# Credit default prediction is central to managing risk in a consumer lending business. Credit default prediction allows lenders to optimize lending decisions, which leads to a better customer experience and sound business economics. Current models exist to help manage risk. But it's possible to create better models that can outperform those currently in use.
#
# In this competition, we’ll apply supervised machine learning to predict credit default. Specifically, we will leverage an industrial scale dataset to build binary classifaction models that challenge the current model in production. Training, validation, and testing datasets include: time-series, behavioral data, and anonymized customer profile information. Apart from creating a base model, we will explore numerous techniques and methodolgies to create an impressive model through feature engineering and using the data in a more organic way within a model.
# ### Objective
# The objective of this competition is to predict the probability that a customer does not pay back their credit card balance amount in the future based on their monthly customer profile. The target binary variable is calculated by observing 18 months performance window after the latest credit card statement, and if the customer does not pay due amount in 120 days after their latest statement date it is considered a default event.
# ### Evaluation Criteria
# If successful, our solution when implemented may yield better customer experiences for cardholders by making it easier for them to be approved for a new credit card. Top solutions may even challenge the credit default prediction model used by the world's largest payment card issuer at American Express.

# ## Data Preprocessing
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

# ### Data Description
# #### Files
# | Filename | Description |
# |-----------------|-----------------------|
# |`train_data.csv`   | training data with multiple statement dates per `customer_ID`|
# |`train_labels.csv` | target label for each `customer_ID`|
# |`test_data.csv`    | corresponding test data; goal: predict `target label` for each `customer_ID`|
# |`sample_submission.csv` | sample submission file in the correct format|
# #### Feature/Target Variables
# The dataset contains aggregated profile features for each customer at each statement date.
#
# Features are anonymized and normalized, and fall into the following general categories:
# | Prefix | Feature Type |
# |:------:|--------------|
# |`D_*`| Delinquency |
# |`S_*`| Spend |
# |`P_*`| Payment |
# |`B_*`| Balance |
# |`R_*`| Risk |
#
# with the following features being categorical:
#     `B_30`, `B_38`, `D_114`, `D_116`, `D_117`, `D_120`, `D_126`, `D_63`, `D_64`, `D_66`, `D_68`
#
#
# **Objective:** For each `customer_ID`, predict the probability of a future payment default (`target == 1`).
#
# **Note:** The negative class (`target == 0`) has been *subsampled at 5%*, and thus *receives a 20x weighting in the scoring metric*.
#
# **Data Source (AMEX):**
#
# American Express is a globally integrated payments company. As the largest payment card issuer in the world, they provide customers with access to products, insights, and experiences that enrich lives and build business success.

# ### Load AMEX Datasets
# **Source (Raw Data):**
# - https://www.kaggle.com/competitions/amex-default-prediction/data
#
# **Source (Compressed Data):**
# - https://www.kaggle.com/datasets/munumbutt/amexfeather
#
# // TODO: Add information regarding the source, advantages and limitations in using the compressed datasets

# +
# train = pd.read_csv("./data/raw/train_data.csv", low_memory=False)
# test = pd.read_csv("./data/raw/test_data.csv")

# +
# Load compressed datasets
# Source: https://www.kaggle.com/datasets/munumbutt/amexfeather

train = pd.read_feather("./data/external/compressed/train_data.ftr")
# test = pd.read_feather("./data/external/compressed/test_data.ftr")
# -

# Preview the first five rows of the training data
train.head()

# Print the shape of the DataFrame for the training set
print(f"Training Data: Shape == {train.shape}")
print(
    f"\nThe training set consists of {train.shape[0]}\
    observations with {train.shape[1]} features\
    and 1 target variable."
)

# Printing the number of observations and features in the dataset.
train.info()

# Preview the first five rows of the testing set
test.head(5)

# Print the shape of the DataFrame for the testing dataset
"""
print(f"Testing Data: Shape == {test.shape}")
print(
    f"\nThe testing set consists of {test.shape[0]} observations with {test.shape[1]} features."
)
"""

# +
# test.info()

# +
# 1-2) Sum the number of incomplete (missing or null) values in each column
# 3-4) Divide by the number of observations and multipy by 100 to make it a percentage.
#   5) Lastly, sort the values in descending order to better observe feature incompleteness.
pct_incomplete = (
    train.isna().sum().div(len(train)).mul(100).sort_values(ascending=False)
)

# Create a set of features which have 20% or more missing values.
incomplete_features = set(pct_incomplete[pct_incomplete >= 20].index)

# +
# Print the number of features with over 20% missing values.
print(
    f"{len(incomplete_features)} features with over 20% values are missing or null.\n"
)

# Print the names of features where 20% or greater have missing or null values
print(f"Incomplete Features: \n{incomplete_features}")
# -

#

# Grouping the data by customer_ID and then counting the number of statements per customer.
statement_counts = (
    train.groupby("customer_ID")
    .size()
    .value_counts(normalize=True)
    .sort_index()
    .reset_index()
    .rename(columns={"index": "n_statements", 0: "pct_customers"})
)

statement_counts

# Plotting a barplot of the number of statements per customer.
sns.barplot(data=statement_counts, x="n_statements", y="pct_customers")

# Creating a list of colors to be used in the pie chart.
colors = sns.color_palette("pastel")
# Creating a figure object with a width of 12 and a height of 12.
fig = plt.figure(figsize=(12, 12))

plt.pie(
    x=statement_counts["pct_customers"][::-1],
    labels=statement_counts["n_statements"][::-1],
    colors=colors,
    autopct="%0.0f%%",
)

# Set the title of the pie chart.
plt.title("Number of Statements per Customer")
# Create a legend for the pie chart.
plt.legend(loc="upper right")
# Display the plot.
plt.show()


# +

# def amex_vars(data=train, select=None, include_target=False, y_col='target'):
# -


def amex_filter(data, feature_type="all", target=False):
    """
    > The function takes a dataframe and returns a subset of it based on the feature type

    Args:
      data: the dataframe to be filtered
      feature_type: str, default 'all'. Defaults to all
      target: if True, the target column will be included in the output. Defaults to False
    """
    # Create a dictionary with the keys being the feature groups and the values
    # being the columns that belong to that group.
    features_dict = {
        "deliquency": data.columns[data.columns.str.startswith("D_")],
        "balance": data.columns[data.columns.str.startswith("B_")],
        "spend": data.columns[data.columns.str.startswith("S_")],
        "risk": data.columns[data.columns.str.startswith("R_")],
        "payment": data.columns[data.columns.str.startswith("P_")],
        "numeric": data.select_dtypes(include="number").columns,
        "categorical": data.select_dtypes(include="category").columns,
        "all": data.columns,
    }

    # Checking if the input is a string. If it is not, raise a TypeError.
    if not isinstance(feature_type, str):
        raise TypeError(
            f"Invalid input - expected str, but acutal: {type(feature_type)}"
        )
    # Check if the feature_type is in features_dict.keys(). If not, raise a ValueError.
    elif feature_type not in features_dict.keys():
        raise ValueError("Invalid feature selection")
    elif target and "target" not in data.columns:
        raise ValueError("Target is not present in dataset!")

    # Retrive list of the columns that are associated with the feature type.
    f_columns = list(features_dict.get(feature_type))

    # Check if keep_target is true and if the target is present in the DataFrame.
    if target and "target" not in f_columns:
        # If not, add it to the DataFrame as a new column.
        f_columns.append("target")
    # If keep_target is false, check if the target is present in the DataFrame.
    # If target column is present, then remove it from the DataFrame.
    elif "target" in f_columns:
        f_columns.remove("target")

    return data.loc[:, f_columns]


amex_filter(train, feature_type="categorical")

# df_deliquency = train.iloc[:, train.columns.str.startswith('D_')]
# df_balance = train.iloc[:, train.columns.str.startswith('B_')]
# df_spend = train.iloc[:, train.columns.str.startswith('S_')]
# df_risk = train.iloc[:, train.columns.str.startswith('R_')]
# df_payment = train.iloc[:, train.columns.str.startswith('P_')]

(
    train.pipe(amex_filter, feature_type="deliquency")
    .pipe(amex_filter, feature_type="categorical")
    .describe()
)

(
    train.pipe(amex_filter, feature_type="deliquency")
    .pipe(amex_filter, feature_type="numeric")
    .describe()
)

(
    train.pipe(amex_filter, feature_type="balance")
    .pipe(amex_filter, feature_type="categorical")
    .describe()
)

(
    train.pipe(amex_filter, feature_type="balance")
    .pipe(amex_filter, feature_type="numeric")
    .describe()
)

df_spend = amex_filter(train, feature_type="spend")
df_spend.describe()

df_risk = amex_filter(train, feature_type="risk")
df_risk.describe()

df_payment = amex_filter(train, feature_type="payment")
df_payment.describe()

train.target.astype("category").describe()

train.target.value_counts(normalize=True)

# +
# last_statement = train.groupby('customer_ID').tail(1).set_index('customer_ID')
# last_statement.target.value_counts(normalize=True)
# -

y_amex = train.set_index("customer_ID").groupby("customer_ID").tail(1).target
X_amex = train.set_index("customer_ID").drop(["S_2", "target"], axis=1)


def aggregate_features(X_amex):
    numeric__agg = (
        X_amex.select_dtypes(include="number")
        .groupby("customer_ID")
        .agg(["first", "last", "mean", "min", "max", "std"])
    )

    categorical__agg = (
        X_amex.select_dtypes(include="category")
        .groupby("customer_ID")
        .agg(["first", "last", "count", "nunique"])
    )

    X_amex__agg = pd.concat([categorical__agg, numeric__agg], axis=1)
    X_amex__agg.columns = ["__".join(col) for col in X_amex__agg.columns]

    return X_amex__agg


X_amex__agg = aggregate_features(X_amex)

from sklearn.model_selection import train_test_split


# amex_numeric__agg = (X_amex
#                      .select_dtypes(include='number')
#                      .groupby('customer_ID')
#                      .agg(['first', 'last', 'mean', 'min', 'max', 'std']))
#
# # amex_numeric__agg.columns = ['__'.join(c) for c in amex_numeric__agg.columns]
# amex_numeric__agg.head(2)

# amex_categorical__agg = (X_amex
#                          .select_dtypes(include='category')
#                          .groupby('customer_ID')
#                          .agg(['first', 'last', 'count', 'nunique']))
#
# # amex_categorical__agg.columns = [
#     #'__'.join(c) for c in amex_categorical__agg.columns]
# amex_categorical__agg.head(2)

# X_agg = pd.concat([amex_categorical__agg, amex_numeric__agg], axis=1)
# X_agg.columns = ['__'.join(col) for col in X_agg.columns]
# X_agg.head(1)

# +
# (X_agg.index == amex_y.index).all()
