import os
import sys
import pickle
import numpy as np 
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import geopy.distance

from src.exception import CustomException
from src.logger import logging

def save_object(file_path: str, obj: any) -> None:
    """
    Saves the given object to the specified file path.

    Args:
        file_path (str): Path to the file where the object needs to be saved.
        obj (any): The object to be saved.

    Raises:
        CustomException: If an exception occurs while saving the object.

    Returns:
        None
    """
    try:
        # Create the directory for the file path if it doesn't exist
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Save the object to the file
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        # Raise a custom exception if an exception occurs while saving the object
        raise CustomException(e, sys)
    

    
def evaluate_model(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, models: dict) -> dict:
    """
    Evaluates the performance of machine learning models using the R2 score.

    Args:
        X_train (np.ndarray: Training features.
        y_train (np.ndarray): Training target.
        X_test (np.ndarray): Testing features.
        y_test (np.ndarray): Testing target.
        models (dict): A dictionary containing machine learning models to be evaluated.

    Returns:
        dict: A dictionary containing the R2 score of each machine learning model.
    """
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train model
            model.fit(X_train, y_train)

            # Predict Testing data
            y_test_pred = model.predict(X_test)

            # Get R2 scores for train and test data
            # train_model_score = r2_score(ytrain,y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        logging.info('Exception occurred during model training')
        raise CustomException(e, sys)
        
        
def load_object(file_path: str) -> any:
    """
    Loads a pickled object from a file.

    Args:
        file_path (str): The path to the pickled object file.

    Returns:
        any: The unpickled object.
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occurred in load_object function utils')
        raise CustomException(e, sys)
        
        
def save_object(file_path: str, obj: any) -> None:
    """
    Saves a pickled object to a file.

    Args:
        file_path (str): The path to save the pickled object file.
        obj (any): The object to be pickled and saved to the file.

    Returns:
        None
    """
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

    
def processing_Date_Time_features(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    A function that processes the raw data by converting the 'Order_Date' column to datetime,
    creating three new features 'Day', 'Month', 'Year', transforming the 'Time_Orderd' and 'Time_Order_picked'
    features into hours with dtype of float.

    Args:
    raw_data: a pandas DataFrame containing the raw data

    Returns:
    processed_data: a pandas DataFrame containing the processed data
    """

    try:
        # chaging dtype of Order_Date to datetime
        raw_data['Order_Date'] = pd.to_datetime(raw_data['Order_Date'], format="%d-%m-%Y")

        # creating three new features order_day, order_month, order_year  
        raw_data['Day'] = raw_data['Order_Date'].dt.day
        raw_data['Month'] = raw_data['Order_Date'].dt.month
        raw_data['Year'] = raw_data['Order_Date'].dt.year

        # droping the column Order_Date since now it has no use
        raw_data = raw_data.drop(labels=['Order_Date'], axis=1)

    except Exception as e:
        logging.info('Exception Occured in process_data function utils')
        raise CustomException(e,sys)

    def time_in_hrs(time: str) -> float:
        """
        A function to convert time in format hrs: min to hour of the data

        Args:
        time: a string value representing the time in the format "hrs: min"

        Returns:
        A float value representing the time in hours.
        """
        # handling nan values
        if time is np.nan:
            return time

        hrs_min = time.split(':')
        # handling values containg only hrs data
        if len(hrs_min) < 2:
            return float(hrs_min[0])

        return float(hrs_min[0]) + float(hrs_min[1])/60

    try:
        # transforming features Time_Orderd and Time_Order_picked
        # we will convert the values of both feature into hours with dtype of float

        ##Fetaure Engineering Process
        raw_data['Time_Orderd'] = raw_data['Time_Orderd'].map(time_in_hrs).astype(float)
        raw_data['Time_Order_picked'] = raw_data['Time_Order_picked'].map(time_in_hrs).astype(float)

        # dropping id
        # processed_data = raw_data.drop(labels=['ID', 'Delivery_person_ID'], axis=1)

    except Exception as e:
        logging.info('Exception Occured in process_data function utils')
        raise CustomException(e,sys)

    return raw_data



##  calculating distance using latitude and longitude using geopy library
def distance(
    restaurant_lat: float, restaurant_lon: float, delivery_lat: float, delivery_lon: float,
) -> float:
    """
    function for calculating distance using latitude and longitude
    """

    return geopy.distance.distance((restaurant_lat, restaurant_lon), (delivery_lat,delivery_lon)).km