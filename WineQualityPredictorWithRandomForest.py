print("Importing required modules")
import numpy as np  # Numerical computaion
import pandas as pd  # Dataframes
from sklearn.model_selection import train_test_split  # Creating training and validation sets
from sklearn import preprocessing  # Pre-processing methods
from sklearn.ensemble import RandomForestRegressor  # Model
from sklearn.pipeline import make_pipeline  # Cross-validation
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import mean_squared_error, r2_score  # Model evaluation
from sklearn.externals import joblib  # Model Persistence

print("Reading wine data from url")
url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
df = pd.read_csv(url, sep=';')  # Reading csv data with ';' separator

print("Records - ")
print(df.head())

print("Shape of the data - ")
print(df.shape)  # (1599, 12) - 1599 observations and 12 features

print("Information about datasets - ")
print(df.describe())

print("Splitting data into training and validation sets - ")
X = df.drop(['quality'],axis=1) # Features 
print("Features shape : ")
print(X.shape)
print("Target shape : ")
Y = df['quality']  # Target
print(Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=123, stratify=Y)  # Creating training and validation sets
print("Training set : ")  # 80% of the datasets
print("Features - ")
print(X_train.shape)
print("Target - ")
print(Y_train.shape)
print("Validation set : ")  # 20% of the datasets
print("Features - ")
print(X_test.shape)
print("Target - ")
print(Y_test.shape)

print("Data pre-processing")
pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=100)) # Setting up pipeline which first standardize the data and the fit random forest regressor on it

print("Setting up hyperparameters")
hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                  'randomforestregressor__max_depth': [None, 5, 3, 1]}

print("Cross validation pipeline")
clf = GridSearchCV(pipeline, hyperparameters, cv = 10) 
clf.fit(X_train, Y_train) # Tuning model using cross-validation pipeline

print("Best params - ")
print(clf.best_params_)
print("Model trained successfully")

print("Prediction - ")
Y_pred = clf.predict(X_test)

print("Mean Squared Error - ")
print(mean_squared_error(Y_test, Y_pred))
print("R2 Score - ")
print(r2_score(Y_test, Y_pred))

print("Saving model")
joblib.dump(clf, 'wine_rf_regressor.pkl')
