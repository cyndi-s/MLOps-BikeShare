#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("../data/train.csv")
data.head()


# ## Data Cleaning

# In[5]:


# Check for missing values
missing_values = data.isnull().sum()
print(missing_values)


# In[6]:


# Box plot to detect outliers in windspeed
plt.boxplot(data['windspeed'])
plt.title('Windspeed Outliers')
plt.show()
# Cap windspeed at a maximum threshold if necessary
data['windspeed'] = data['windspeed'].clip(upper=40)


# In[7]:


# Remove rows with negative or erroneous values
data = data[data['temp'] >= 0]


# ## Feature Engineering

# In[8]:


# Convert 'datetime' column to datetime object
data['datetime'] = pd.to_datetime(data['datetime'])

# Create new features
data['hour'] = data['datetime'].dt.hour
data['day_of_week'] = data['datetime'].dt.dayofweek
data['month'] = data['datetime'].dt.month


# In[9]:


# Create binary weather features
data['is_clear_weather'] = (data['weather'] == 1).astype(int)
data['is_rainy_weather'] = (data['weather'] >= 3).astype(int)


# In[10]:


# Create a combined feature for holidays and working days
data['is_holiday_workingday'] = ((data['holiday'] == 1) & (data['workingday'] == 1)).astype(int)


# In[11]:


data.drop(columns=["datetime"], inplace=True)


# ## Data Split

# In[12]:


from sklearn.model_selection import train_test_split
# Split the data into features and target
X = data.drop(columns=["count"])  # Features (all columns except 'count')
y = data["count"]  # Target variable
# Perform an 80-20 train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Verify the split
print(f"Training data size: {X_train.shape}")
print(f"Testing data size: {X_test.shape}")


# ## Starting MLFlow run and Logging

# In[13]:


import mlflow
import mlflow.sklearn
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np


# In[14]:


# Define the model 
model = DecisionTreeRegressor(max_depth=10, random_state=42)

# Start an MLflow run 
with mlflow.start_run():
    # Log model parameters
    mlflow.log_param("model_type", "DecisionTreeRegressor")
    mlflow.log_param("max_depth", 10)
    #Train the Model 
    model.fit(X_train, y_train)
    #Make predicition
    predictions = model.predict(X_test)
    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    #Log evaluation metrics 
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    # Plot and log feature importance as an artifact
    feature_importances = model.feature_importances_
    plt.figure(figsize=(10, 6))
    plt.barh(X_train.columns, feature_importances)
    plt.title("Feature Importance")
    plt.savefig("feature_importance.png")
    #Log artifacts(the feature importance plot)
    mlflow.log_artifact("feature_importance.png")
    #Log the model itself 
    mlflow.sklearn.log_model(model,"model")

print(f"Model training complete. MAE: {mae}, RMSE: {rmse}")


# In[15]:


print(mlflow.get_tracking_uri())


# ## HyperTune Parameters - Log all the parameters

# In[16]:


from itertools import product
from sklearn.model_selection import GridSearchCV

model = DecisionTreeRegressor(random_state=42)
param_grid = {
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 10, 20]
}

with mlflow.start_run(run_name= "GridSearch for Regressor"):
    # Set up the gridSearchCV
    gridSearch = GridSearchCV(estimator=model, param_grid=param_grid, cv=5,
                               scoring='neg_mean_squared_error', verbose=1, return_train_score=True)
    gridSearch.fit(X_train, y_train)

    for i,param in enumerate(gridSearch.cv_results_['params']):
        meanScore = -gridSearch.cv_results_['mean_test_score'][i]
        stdScore = gridSearch.cv_results_['std_test_score'][i]

        with mlflow.start_run(run_name=f"Params {i+1}", nested=True):
            mlflow.log_params(param)
            mlflow.log_metric("mean_cv_rmse", meanScore)
            mlflow.log_metric("sd_cv_rmse", stdScore)

    best_params = gridSearch.best_params_
    best_score = -gridSearch.best_score_

    mlflow.log_params(best_params)
    mlflow.log_metric("best_cv_rmse", best_score)

    best_model = gridSearch.best_estimator_
    mlflow.sklearn.log_model(best_model, "best_model")

    test_predictions = best_model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    mlflow.log_metric("test_RMSE", test_rmse)

print(f"Best Hyperparameters: {best_params}")
print(f"Test RMSE: {test_rmse}")


# ## Logging Metrics

# In[17]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow


# In[19]:


#Evaluate the model
y_pred = best_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)


# In[20]:


# log Metrics using MLFLOW 
with mlflow.start_run():
    mlflow.log_metric("R2", r2)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("MAE", mae)


# In[21]:


print(f"R²: {r2:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")


# ## Model Deployment

# In[25]:


# Save the model 
with mlflow.start_run():
    inputExample = X_train.iloc[[0]]
    mlflow.sklearn.log_model(best_model, "bikePredModel", input_example=inputExample)
    print("Model saved")


# In[26]:


## The model is then served Locally and is tested using Curl Commands


# In[ ]:




