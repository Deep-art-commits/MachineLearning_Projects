import os
import sys
from src.exception import custom_exception
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass # used to define class variables without using init
from src.components.data_transformation import Data_transformation
from src.components.data_transformation import Data_transformation_config
from src.components.model_trainer import Model_trainer

@dataclass
class Data_ingestion_config:
    raw_data_path   :str   =    os.path.join('artifacts','raw.csv')
    train_data_path :str   =    os.path.join('artifacts','train.csv')
    test_data_path  :str   =    os.path.join('artifacts','test.csv')

class Data_ingestion:
    def __init__(self):
        self.ingestion_config=Data_ingestion_config()
    def initiate_data_ingestion(self):
        logging.info("Entered the Data_ingestion_method")
        try:
            df=pd.read_csv("notebook\data\stud.csv")
            logging.info("Reading   dataset as pandas dataframe")
            logging.info("Creating Directories for raw data")
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Raw_file _created :")
            
            logging.info("Train _test split started")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=28)
            logging.info("Train _test split Completed ")
            
            logging.info("Creating Directories for train data")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            logging.info("train_file _created :")
            logging.info("Creating Directories for test data")
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path),exist_ok=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("test_file _created :")
            
            logging.info("Data ingestion Complete:")
            
            return(
                self.ingestion_config.test_data_path,
                self.ingestion_config.train_data_path
            )
            
            
        except Exception as e:
            raise custom_exception(e,sys)  
        
if __name__=="__main__":
    obj=Data_ingestion()
    test_data,train_data=obj.initiate_data_ingestion()
    data_transformation=Data_transformation()
    train_array,test_array,_=data_transformation.initiate_data_transformation(train_data,test_data)
    Model_Trainer=Model_trainer()
    print(Model_Trainer.initiate_model_training(train_array,test_array))
    