import os
import time
import json
import joblib
import mlflow
import numpy as np
import pandas as pd
import mlflow.sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from google.oauth2 import service_account
from mlflow.models import infer_signature
from mlflow_sdk.gcp_functions import download_files
from mlflow_sdk.gcp_functions import upload_files
from mlflow_sdk.train_predict_abj import model_training
from mlflow_sdk.train_predict_lag import model_training_
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def experiment_settings():
    run_name = str(int(time.time()))
    print('Run name: ', run_name)

    # set the tracking uri
    artifactPath="default"
    mlflow_experiment_id=0
    server_uri='http://127.0.0.1:5000'
    mlflow.set_tracking_uri(server_uri)
    mlflow.set_experiment('demand forecast')
    experiment=mlflow.get_experiment('0')

    print("Name of experiment: {}".format(experiment.name))
    print("Location of Artifact: {}".format(experiment.artifact_location))
    print("Life cycle phase: {}".format(experiment.lifecycle_stage))
    print("Experiment_ID: {}".format(experiment.experiment_id))

    directory1 = "./data/dict"
    directory2 = './data/processed/'
    directory3 = './data/raw/'
    directory4 = './models/'
        
    if not os.path.exists(directory1):
        os.makedirs(directory1)
        
    if not os.path.exists(directory2):
        os.makedirs(directory2)

    if not os.path.exists(directory3):
        os.makedirs(directory3)

    if not os.path.exists(directory4):
        os.makedirs(directory4)

    download_files("df-ml-models", "./data/dict", "dict")

    return run_name, mlflow_experiment_id

# Train the model
def train_model_abj(runName, mlflow_experiment_id):     

    # Create or Select Experiment 
    experiment = mlflow.set_experiment(runName)
    print("Name of experiment: {}".format(experiment.name))
    with mlflow.start_run(experiment_id=mlflow_experiment_id):      
        mf, md, lr, X_train, X_test, y_train, y_test = model_training()    
        # train algorithm with the best parameters
        randomState = 0
        GB_cen = GradientBoostingRegressor(n_estimators=200,max_features=mf,max_depth=md,learning_rate=lr,random_state=randomState).ﬁt(X_train,y_train)
        oos_r2 = r2_score(y_test, GB_cen.predict(X_test))
        rsme = np.sqrt(mean_squared_error(y_test, GB_cen.predict(X_test)))
        print('OOS R2:', oos_r2)
        print("rsme:", rsme)
        signature = infer_signature(X_test, GB_cen.predict(X_test))
        # log tag
        mlflow.set_tag(key='Location', value='abuja models')
        # Log Parameters & Metrics
        mlflow.log_params({"max_features": mf, "max_depth": md, "learning_rate": lr, "random state": randomState})        
        mlflow.log_metrics({"rsme": rsme, "r2_score": oos_r2})
        # Log Model
        mlflow.sklearn.log_model(GB_cen, "model", signature=signature, registered_model_name="sklearn-gboost-reg-model-abj",)
        # log artifacts
        mlflow.log_artifacts("./data")
        # save the model to disk
        filename = './models/abj_model.sav'
        joblib.dump(GB_cen, filename) # to gcloud bucket
        # End tracking
        mlflow.end_run()
        print("finished training")

# Train the model
def train_model_lag(runName, mlflow_experiment_id):     

    # Create or Select Experiment 
    experiment = mlflow.set_experiment(runName)
    print("Name of experiment: {}".format(experiment.name))
    with mlflow.start_run(experiment_id=mlflow_experiment_id):      
        mf, md, lr, X_train, X_test, y_train, y_test = model_training_()
        # train algorithm with the best parameters
        randomState = 0
        GB_cen = GradientBoostingRegressor(n_estimators=200, max_features=mf,max_depth=md,learning_rate=lr,random_state=0).ﬁt(X_train,y_train)
        oos_r2 = r2_score(y_test, GB_cen.predict(X_test))
        rsme = np.sqrt(mean_squared_error(y_test, GB_cen.predict(X_test)))
        print('OOS R2:', oos_r2)
        print("rsme:", rsme)
        signature = infer_signature(X_test, GB_cen.predict(X_test))
        # log tag
        mlflow.set_tag(key='Location', value='lagos models')
        # Log Parameters & Metrics
        mlflow.log_params({"max_features": mf, "max_depth": md, "learning_rate": lr, "random state": randomState})        
        mlflow.log_metrics({"rsme": rsme, "r2_score": oos_r2})
        # Log Model
        mlflow.sklearn.log_model(GB_cen, "model", signature=signature, registered_model_name="sklearn-gboost-reg-model-lag",)
        # save the model to disk
        filename = './models/lag_model.sav'
        joblib.dump(GB_cen, filename) # to gcloud bucket
        # End tracking
        mlflow.end_run()
        print("finished training")

if __name__ == "__main__":
    run_name, mlflow_experiment_id = experiment_settings()
    train_model_abj(run_name, mlflow_experiment_id)#
    time.sleep(1)
    train_model_lag(run_name, mlflow_experiment_id)
    time.sleep(1)
    upload_files(bucket_name="df-ml-models", local_directory="./models")