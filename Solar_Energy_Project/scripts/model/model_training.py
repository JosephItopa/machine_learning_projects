# import external modules

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set() # Revert to matplotlib defaults
plt.rcParams['figure.figsize'] = (16, 12)

# import libraries for algorithms traininng, and metrics to judge performance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# import local modules
from data_cleaning.datacleaning import clean_data

# clean and split dataset to train, test, and validation dataset
train_data, test_data, validation_data = clean_data()

# training data
# train_data = pd.read_csv('~/Documents/Projects/update_project/My_Best_Projects/Solar_Energy_Project/datasets/cleaned/train.csv')
X_train = train_data.drop(['Daily_radiation'], axis = 1)
y_train = train_data['Daily_radiation']

# testing data
# test_data = pd.read_csv('~/Documents/Projects/update_project/My_Best_Projects/Solar_Energy_Project/datasets/cleaned/test.csv')
X_test = test_data.drop(['Daily_radiation'], axis = 1)
y_test = test_data['Daily_radiation']

# validation data
# validation_data = pd.read_csv('~/Documents/Projects/update_project/My_Best_Projects/Solar_Energy_Project/datasets/cleaned/validation.csv')
X_val = validation_data.drop(['Daily_radiation'], axis=1)
y_val = validation_data['Daily_radiation']

# Setup the pipeline steps for linear regression
steps_lr = [('scaler', StandardScaler()), ('lr', LinearRegression())]
# Create the pipeline
pipeline_lr = Pipeline(steps_lr)

# Setup the pipeline steps for random forest: steps
steps_rf = [('scaler', StandardScaler()), ('rfr', RandomForestRegressor())]
# Create the pipeline: pipeline
pipeline_rfr = Pipeline(steps_rf)

# Setup the pipeline steps: steps
steps_gbr = [('scaler', StandardScaler()), ('gbr', GradientBoostingRegressor())]
# Create the pipeline: pipeline
pipeline_gbr = Pipeline(steps_gbr)

def model_train_test(X_train, y_train, X_test, y_test):

    # Fit the pipeline to the train set
    pipeline_lr.fit(X_train, y_train)
    # Predict the labels of the test set
    y_pred_lr = pipeline_lr.predict(X_test)
    # Evaluating algorithm performance
    mse = mean_squared_error(y_test, y_pred_lr, squared = False)

    mae = mean_absolute_error(y_test, y_pred_lr)

    print('r2_score: ', r2_score(y_test, y_pred_lr))

    print('Root Mean Squared Error: %.2f' % np.sqrt(mse))

    print('Root Mean Absolute Error: %.2f' % np.sqrt(mae))


    # Fit the pipeline to the train set using randomforest regressor
    pipeline_rfr.fit(X_train, y_train)
    # Predict the labels of the test set
    y_pred_rfr = pipeline_rfr.predict(X_test)
    # Evaluating algorithm performance
    mse_rf = mean_squared_error(y_test, y_pred_rfr, squared = False)

    mae_rf = mean_absolute_error(y_test, y_pred_rfr)

    print('r2_score: ', r2_score(y_test, y_pred_rfr))

    print('Root Mean Squared Error: %.2f' % np.sqrt(mse_rf))

    print('Root Mean Absolute Error: %.2f' % np.sqrt(mae_rf))


    # Fit the pipeline to the train set using gradientboost regressor
    pipeline_gbr.fit(X_train, y_train)
    # Predict the labels of the test set
    y_pred_gbr = pipeline_gbr.predict(X_test)
    # Evaluating algorithm performance
    mse_gr = mean_squared_error(y_test, y_pred_gbr, squared = False)

    mae_gr = mean_absolute_error(y_test, y_pred_gbr)

    print('r2_score: ', r2_score(y_test, y_pred_gbr))

    print('Root Mean Squared Error: %.2f' % np.sqrt(mse_gr))

    print('Root Mean Absolute Error: %.2f' % np.sqrt(mae_gr))


def model_validation(X_val, y_val):
    # validate models
    y_val_lr = pipeline_lr.predict(X_val)
    y_val_rfr = pipeline_rfr.predict(X_val)
    y_val_gbr = pipeline_gbr.predict(X_val)


    # Evaluating algorithm performance for linear regression
    mse_lr_val = mean_squared_error(y_val, y_val_lr, squared = False)

    mae_lr_val = mean_absolute_error(y_val, y_val_lr)

    print('r2_score: ', r2_score(y_val, y_val_lr))

    print('Linear Regression - Root Mean Squared Error: %.2f' % np.sqrt(mse_lr_val))

    print('Linear Regression - Root Mean Absolute Error: %.2f' % np.sqrt(mse_lr_val))


    # Evaluating algorithm performance for random forest regression
    mse_rf_val = mean_squared_error(y_val, y_val_rfr, squared = False)

    mae_rf_val = mean_absolute_error(y_val, y_val_rfr)

    print('r2_score: ', r2_score(y_val, y_val_rfr))

    print('Random Forest - Root Mean Squared Error: %.2f' % np.sqrt(mse_rf_val))

    print('Random Forest - Root Mean Absolute Error: %.2f' % np.sqrt(mse_rf_val))


    # Evaluating algorithm performance for gradient boost regression
    mse_gbr_val = mean_squared_error(y_val, y_val_gbr, squared = False)

    mae_gbr_val = mean_absolute_error(y_val, y_val_gbr)

    print('r2_score: ', r2_score(y_val, y_val_gbr))

    print('Gradient Boost - Root Mean Squared Error: %.2f' % np.sqrt(mse_gbr_val))

    print('Gradient Boost - Root Mean Absolute Error: %.2f' % np.sqrt(mae_gbr_val))

