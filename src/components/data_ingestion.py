# Reading the data from data source, split, and eventually save the data

import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

# Define a data class to hold the configuration for data ingestion
@dataclass
class DataIngestionConfig:
    """ It gives any path that I will require"""
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

# Define the DataIngestion class
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Load the data from CSV file into a Pandas DataFrame
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info("Read the dataset as dataframe")

            # Create the 'artifacts' folder if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw data as CSV
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved")

            # Split the data into train and test sets
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save the train set as CSV
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            logging.info("Train data saved")

            # Save the test set as CSV
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Test data saved")

            logging.info("Ingestion of the data is completed")

            # Return the paths to the train and test data CSV files
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            # If any error occurs during data ingestion, raise a custom exception
            raise CustomException(e, sys)

# The main block of the script
if __name__ == "__main__":
    # Create an instance of DataIngestion
    obj = DataIngestion()

    # Initiate the data ingestion process and get the paths to the train and test data CSV files
    train_data, test_data = obj.initiate_data_ingestion()

    # Create an instance of DataTransformation and initiate data transformation on the ingested data
    data_transformation = DataTransformation()
    train_arr, test_arr,_ = data_transformation.initiate_data_transformation(train_data, test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))
