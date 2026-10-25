import sklearn
import pandas as pd
from sklearn import ensemble
from sklearn import metrics
import joblib
import numpy as np

import dagshub

dagshub.init(repo_owner='ftayebi351', repo_name='Template_MLOps_accidents', mlflow=True)

import mlflow

print(joblib.__version__)

X_train = pd.read_csv('data/preprocessed/X_train.csv')
X_test = pd.read_csv('data/preprocessed/X_test.csv')
y_train = pd.read_csv('data/preprocessed/y_train.csv')
y_test = pd.read_csv('data/preprocessed/y_test.csv')
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

rf_classifier = ensemble.RandomForestClassifier(n_estimators=200, criterion="entropy")

#--Train the model and log to MLflow
with mlflow.start_run():
    mlflow.log_param('n_estimators', 200)
    mlflow.log_param('criterion', 'entropy')
    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    mlflow.log_metric('accuracy', accuracy)

#--Save the trained model to a file
model_filename = './models/trained_model.joblib'
joblib.dump(rf_classifier, model_filename)
print("Model trained and saved successfully.")
