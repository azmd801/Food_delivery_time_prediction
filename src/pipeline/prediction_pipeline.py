import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object,processing_Date_Time_features,distance
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
    def __init__(self,**kwargs):
  
        self.datapoint = kwargs

    def get_data_as_dataframe(self) -> pd.DataFrame:
        """
        Returns a pandas dataframe with the provided features and data point

        Returns:
        - pd.DataFrame: dataframe containing data point with provided features
        """
        try:
            df = pd.DataFrame(data=[self.datapoint.values()], columns=self.datapoint.keys() )
            logging.info('Dataframe Gathered')
            # logging.info(df.head())

            # Applying some transformation on Gathered data frame for prepreceoosr pickle object to read
            logging.info('Applying some transformation on Gathered data frame for prepreceoosr pickle object to read')
            df = processing_Date_Time_features(df)
            df['restaurant_delivery_location_dist'] = df.apply(lambda row: distance(row.Restaurant_latitude, row.Restaurant_longitude, \
                                                           row.Delivery_location_latitude,row.Delivery_location_longitude),axis=1)
            
            df = df.drop(['Month','Year'],axis=1)

            return df

        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e, sys)