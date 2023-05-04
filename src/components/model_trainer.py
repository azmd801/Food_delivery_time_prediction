# Import necessary libraries and modules
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

# Import custom modules
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

from dataclasses import dataclass
import sys
import os

# Define dataclass for model trainer configuration
@dataclass
class ModelTrainerConfig:
    """
    A dataclass to hold the configuration for the ModelTrainer class.

    Attributes:
    -----------
    trained_model_file_path : str
        The path to save the trained model file.

    """

    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

    


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self, train_array: np.ndarray, test_array: np.ndarray) -> None:
        """
        Trains the best machine learning model based on the train and test arrays

        Args:
            train_array (np.ndarray): Array containing the training data
            test_array (np.ndarray): Array containing the test data
        """
        try:
            # Split dependent and independent variables from train and test data
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Create a dictionary of models to be evaluated
            models = {
                'LinearRegression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'Elasticnet': ElasticNet()
            }

            # Evaluate the models and get the model report
            model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # Get the best model from the model report
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            # Save the best model
            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

        except Exception as e:
            logging.info('Exception occurred at Model Training')
            raise CustomException(e, sys)