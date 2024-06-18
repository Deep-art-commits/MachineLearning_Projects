import os
import sys
from src.exception import custom_exception
from src.logger import logging
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from dataclasses import dataclass # used to define class variables without using init
from src.utils import save_object

@dataclass
class Data_transformation_config:
    preprocessor_file_obj=os.path.join('artifacts',"preprocessor.pkl")
    
class Data_transformation:   
    def __init__(self):
        self.data_transformation_config=Data_transformation_config()
    def get_datatransformer_obj(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            logging.info("Reading numerical and categorical features")
            categorical_columns=['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            numerical_columns=['reading_score', 'writing_score']
            logging.info("Creating pipeline for numerical variables")
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            logging.info("Creating pipeline for categorical variables")
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("encode",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            
            logging.info("Combining pipelines for numerical and categorical variables")
            preprocessor=ColumnTransformer(
                [
                    ("numerical_pipeline",num_pipeline,numerical_columns),
                    ("categorical_pipeline",cat_pipeline,categorical_columns)
                ]
            )
            logging.info("Encoding complete for numerical and categorical variables")
            
            return preprocessor
            
        except Exception as e:
            raise custom_exception(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test dataset")
            
            logging.info("Obtaining preprocessor object")
            
            preprocessing_obj=self.get_datatransformer_obj()
            
            target_column_name="math_score"
            numerical_columns=['reading_score', 'writing_score']
            
            input_features_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_features_train_df=train_df[target_column_name]
            
            input_features_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_features_test_df=test_df[target_column_name]
            
            logging.info("Applying Preprocessing")
            input_features_train_arr=preprocessing_obj.fit_transform(input_features_train_df)
            input_features_test_arr=preprocessing_obj.transform(input_features_test_df)
            
            train_arr=np.c_[input_features_train_arr,np.array(target_features_train_df)]
            test_arr=np.c_[input_features_test_arr,np.array(target_features_test_df)]
            
            logging.info("saving preprocessing object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_file_obj,
                obj=preprocessing_obj
            )
            
            
            
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_file_obj
            )
        except Exception as e:
            raise custom_exception(e,sys)
                