import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
#  Modelling 
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
    )
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from dataclasses import dataclass
from src.utils import save_object,evaluate_model
from src.exception import custom_exception
from src.logger import logging
import os
import sys
import warnings

@dataclass
class Model_trainer_config:
    trained_model_filepath=os.path.join('artifacts',"model.pkl")
class Model_trainer:
    def __init__(self):
        self.model_trainer_config=Model_trainer_config()
    def initiate_model_training(self,train_array,test_array):
        
        try:
            logging.info("Model Training Started")  
            logging.info("Splitting train and test from input")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)
            
        
            # to get best modelscore
            best_model_score=max(sorted(model_report.values()))
            # to get  model name for best modelscore
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            best_model=models[best_model_name]
            if best_model_score<0.6:
                print("No best model found")
            logging.info("Best model found on both traininga nd test DS")    
            
            save_object(
                file_path=self.model_trainer_config.trained_model_filepath,
                obj=best_model
            ) 
            
            predicted=best_model.predict(X_test)
            score=r2_score(y_test,predicted)
            adj_rscore=1-(1-score)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
            return best_model, score,adj_rscore
        
        
        except Exception as e:
            raise custom_exception(e,sys) 