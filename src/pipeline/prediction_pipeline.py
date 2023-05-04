import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd
import numpy as np


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features: pd.DataFrame) -> np.array:
        """
        Takes in input dataframe and predicts the target variable

        Args:
        - features (pd.DataFrame): Input data

        Returns:
        - np.array: Predicted output
        """
        try:
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)
            return pred

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, features: list, data_point: tuple):
        """
        Initialize CustomData class instance

        Args:
        - features (list): list of feature names
        - data_point (tuple): tuple of data values
        """
        self.features = features
        self.data_point = data_point

    def get_data_as_dataframe(self) -> pd.DataFrame:
        """
        Returns a pandas dataframe with the provided features and data point

        Returns:
        - pd.DataFrame: dataframe containing data point with provided features
        """
        try:
            df = pd.DataFrame([self.data_point], columns=self.features)
            logging.info('Dataframe Gathered')
            return df

        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e, sys)