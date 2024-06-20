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
from xgboost import XGBRFRegressor
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
            models={
                "LinearRegression":LinearRegression(),
                "Lasso":Lasso(),
                "Ridge":Ridge(),
                "KNeighborsRegressor":KNeighborsRegressor(),
                "DecisionTree":DecisionTreeRegressor(),
                "RandomForest":RandomForestRegressor(),
                "XGBRFRegressor":XGBRFRegressor(),
                # "CatBoost":CatBoostRegressor(verbose=False),
                "AdaBoost":AdaBoostRegressor(),
                "Gradient_Boost":GradientBoostingRegressor()
            }
            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            
        
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
            # adj_rscore=1-(1-score)*(len(y_test)-1))/(len(y_test)-X_test.shape()[1]-1
            return best_model, score
        
        
        except Exception as e:
            raise custom_exception(e,sys) 