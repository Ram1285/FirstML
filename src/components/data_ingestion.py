import os 
import sys
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exception import CustomException

@dataclass
class DataIngestionConfig:
    test_data_path:str=os.path.join('artifacts','test.csv')
    train_data_path:str=os.path.join('artifacts','train.csv')
    raw_data_path:str=os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.data_ingestion=DataIngestionConfig()
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")

        try:
            df=pd.read_csv(r'notebooks\dataset\stud.csv')
            logging.info("Read the data as dataframe")
            os.makedirs(os.path.dirname(self.data_ingestion.train_data_path),exist_ok=True)
            df.to_csv(self.data_ingestion.raw_data_path,header=True,index=False)
            logging.info("Initiated train test split")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.data_ingestion.train_data_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion.test_data_path,index=False,header=True)
            logging.info("Ingestion of data completed")
            return(
                self.data_ingestion.train_data_path,
                self.data_ingestion.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()