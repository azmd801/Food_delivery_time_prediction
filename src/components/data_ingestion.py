import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from typing import Tuple

from src.components.data_transformation import DataTransformation


## Intitialize the Data Ingetion Configuration

@dataclass
class DataIngestionconfig:
    """Configuration class for data ingestion.

    This class provides default values for the paths to the training, testing,
    and raw data files. These paths can be overridden by providing new values
    when creating an instance of the class.

    Attributes:
        train_data_path (str): The path to the training data file.
        test_data_path (str): The path to the testing data file.
        raw_data_path (str): The path to the raw data file.
    """
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')


## create a class for Data Ingestion
class DataIngestion:
    def __init__(self):
        """
        Initializes the DataIngestion object with default configuration values.
        """
        self.ingestion_config = DataIngestionconfig()

    def initiate_data_ingestion(self) -> Tuple[str, str]:
        """
        Performs data ingestion by reading the finalTrain.csv file, saving a copy
        of the raw data to disk, and splitting the data into training and testing sets.
        Returns the paths to the training and testing data files.

        Returns:
            A tuple containing the paths to the training and testing data files.
        """
        logging.info('Data Ingestion methods Starts')
        try:
            df = pd.read_csv(os.path.join('Notebook', 'data', 'finalTrain.csv'))
            logging.info('Dataset read as pandas Dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info('Raw data saved to disk at %s', self.ingestion_config.raw_data_path)

            logging.info('Train test split')
            train_set, test_set = train_test_split(df, test_size=0.30, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Data ingestion completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error('Error occurred during Data Ingestion: %s', str(e))
            raise CustomException(e, sys)
