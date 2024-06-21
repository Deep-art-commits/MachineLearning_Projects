import os 
import sys
from src.exception import custom_exception
import numpy as np
import pandas as pd
import dill
from src.logger import logging
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as f_obj:
            dill.dump(obj,f_obj)

    except Exception as e:
                raise custom_exception(e,sys)  

def evaluate_model(X_train,y_train,X_test,y_test,models,param):
    try:
        reportpath :str=os.path.join('artifacts',"metricsreport.csv")
        os.makedirs(os.path.dirname(reportpath),exist_ok=True)
        report={}
        report_d={}
        model_lst=[]
        for i in range(len(list(models))):
            model=list(models.values())[i]
            para=param[list(models.keys())[i]]
            
            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)
            
            logging.info("Model Evaluation started")
            # logging.info(f"Evaluating data using {model[i]}: ")
            model.fit(X_train,y_train)
            y_train_pred=model.predict(X_train)
            y_test_pred=model.predict(X_test)
            Train_model_Rscore=r2_score(y_train,y_train_pred)
            Test_model_Rscore=r2_score(y_test,y_test_pred)
            Train_model_adj_Rscore=1-(1-Train_model_Rscore)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
            Test_model_adj_Rscore=1-(1-Test_model_Rscore)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
            report[list(models.keys())[i]]=Test_model_adj_Rscore
            # report_d[list(models.keys())[i]]=[{'Train_model_Rscore':Train_model_Rscore,
            #                                    'Test_model_Rscore':Test_model_Rscore,
            #                                    'Train_model_adj_Rscore':Train_model_adj_Rscore,
            #                                    'Test_model_adj_Rscore':Test_model_adj_Rscore}]
            report_d[list(models.keys())[i]]=[Train_model_Rscore,
                                               Test_model_Rscore,
                                               Train_model_adj_Rscore,
                                               Test_model_adj_Rscore]
            
            report_df=pd.DataFrame(list(report_d.items()),columns=['key','values'],)
            
            
            report_df.to_csv(reportpath,header=True)
            # print(report_d,report_df)
            
        return report
        print(report_d)
    except Exception as e:
                raise custom_exception(e,sys) 
def load_object(file_path)    :
    try:
        with open(file_path,"rb") as f_obj:
            return dill.load(f_obj) 
    except Exception as e:
                raise custom_exception(e,sys)           
        