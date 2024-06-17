import os
import sys
from src.exception import custom_exception
from src.logger import logging
import pandas as pd 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import Data_transformation
from src.components.data_transformation import Data_transformation_config


@dataclass
class Data_ingestion_config:
    raw_data_path:str=os.path.join('artifacts',"raw_data.csv")
    train_data_path:str=os.path.join('artifacts',"train.csv")
    test_data_path:str=os.path.join('artifacts',"test.csv")

class data_ingestion:
    def __init__(self) :
        self.ingestion_config=Data_ingestion_config()
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('notebook\data\stud.csv')
            logging.info("Reading the dataset as a Pandas Dataframe ")
             # creating directories
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
           
            logging.info("Train_test_split started ")
            # splitting data into test and train 
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=10)
            # Storing train and test data in respective csv files 
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Ingestion of Data is complete ")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e :
            raise custom_exception(e,sys)

if __name__=="__main__" :
    obj=   data_ingestion() 
    train_data,test_data=obj.initiate_data_ingestion()
    data_transformation=Data_transformation()
    data_transformation.initiate_data_transformation(train_data,test_data)
   