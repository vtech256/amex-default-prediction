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
