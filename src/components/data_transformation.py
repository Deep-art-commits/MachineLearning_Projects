import sys
import os 
from dataclasses import dataclass
import numpy as np
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
# For Categorical variables use one hot encoder and for numerical values use Standard scler .
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.exception import custom_exception
from src.logger import logging
from src.utils import save_object
from src.components.data_ingestion import data_ingestion
from src.components.data_ingestion import Data_ingestion_config

@dataclass
class Data_transformation_config:
    preprocessor_ob_file_path=os.path.join('artifacts','preprocessor.pkl')
class Data_transformation:
    def __init__(self):
        self.Data_transformation_config=Data_transformation_config()
    def get_data_transformer_obj(self):
        '''
        This function does data transformation 
        '''
        try:
            num_vars=[ 'reading_score', 'writing_score']
            cat_vars=['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            num_pipeline=Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
            ]
                                  )
            logging.info("Standard Scaling for numerical columns encoding completed ")
            Categirical_pipeline=Pipeline(steps=[
            ('imputer',SimpleImputer(strategy='most_frequent')),
            ('one_hot_encoder',OneHotEncoder()),
            ('scaler',StandardScaler())
                    
                ]
            )
            logging.info("Categorical columns encoding completed ")
            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,num_vars),
                ("Categirical_pipeline",Categirical_pipeline,cat_vars)]
                )
            return preprocessor
        except Exception as e:
            raise custom_exception(e,sys) 
        
    def initiate_data_transformation(self,train_data_path,test_data_path):
            try:
                train_df=pd.read_csv(train_data_path)
                test_df=pd.read_csv(test_data_path)
                logging.info("Test and train Dataset read")
                
                logging.info("Getting Preprocessor object")
                preprocessor_obj=self.get_data_transformer_obj()
                target_column="math_score"
                num_vars=[ 'reading_score', 'writing_score']
                input_feature_train_df=train_df.drop(columns=[target_column],axis=1)
                target_feature_train_df=train_df[target_column]
                
                input_feature_test_df=test_df.drop(columns=[target_column],axis=1)
                target_feature_test_df=test_df[target_column]
                
                input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
                input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)
                
                
                logging.info(" Applying Preprocessing  on train and test ")
                
                
                train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
                test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
                
                logging.info("Saving preprocessing object")
                
                save_object(
                    file_path=self.Data_transformation_config.preprocessor_ob_file_path,
                    obj=preprocessor_obj
                )
                return(
                    train_arr,
                    test_arr,
                    self.Data_transformation_config.preprocessor_ob_file_path
                )
            except Exception as e:
                raise custom_exception(e,sys)    
            
            
            
           

            