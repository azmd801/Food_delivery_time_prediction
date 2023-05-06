import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from pandas.core.indexes.base import Index

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object
from src.utils import processing_Date_Time_features
from src.utils import distance

@dataclass
class DataTransformationConfig:
    """Configuration class for data transformation.

    This class provides a default value for the file path to the preprocessor
    object, which is used to preprocess the data before training.

    Attributes:
        preprocessor_obj_file_path (str): The file path to the preprocessor object.
    """
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    """
    Class to transform data for the model training
    """

    def __init__(self):
        """
        Initialize DataTransformation object
        """
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self, numerical_cols: Index, categorical_cols: Index) -> ColumnTransformer:
        """
        Get a preprocessor object for data transformation
        
        Args:
        numerical_cols: Index object containing names of numerical columns
        categorical_cols: Index object containing names of categorical columns
        
        Returns:
        A ColumnTransformer object for data transformation
        """
        try:
            logging.info("Data Transformation initiated")
            logging.info("Pipeline Initiated")

            # Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            # Categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_cols),
                    ("cat_pipeline", cat_pipeline, categorical_cols),
                ]
            )

            logging.info("Pipeline Completed")

            return preprocessor

        except Exception as e:
            logging.error("Error in Data Transformation")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str) -> tuple:
        """
        Read data from given file paths, transform the data and return the processed data
        
        Args:
        train_path: File path of the training data
        test_path: File path of the testing data
        
        Returns:
        Tuple containing processed training and testing data, preprocessor pickle file path
        """

        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info(f"Train Dataframe Head: \n{train_df.head().to_string()}")
            logging.info(f"Test Dataframe Head: \n{test_df.head().to_string()}")

            # processing features features Order_Date_Time, Orderd_Time and Order_picked
            logging.info("Processing features: Order_Date_Time, Orderd_Time and Order_picked")

            train_df = processing_Date_Time_features(train_df)
            test_df = processing_Date_Time_features(test_df)

            # adding a new feature 'restaurant_delivery_location_dist' in the data using latitude and logitude features
            logging.info("adding a new feature 'restaurant_delivery_location_dist' in the data using latitude and logitude features")

            train_df['restaurant_delivery_location_dist'] = train_df.apply(lambda row: distance(row.Restaurant_latitude, row.Restaurant_longitude, \
                                                           row.Delivery_location_latitude,row.Delivery_location_longitude),axis=1)
            
            test_df['restaurant_delivery_location_dist'] = test_df.apply(lambda row: distance(row.Restaurant_latitude, row.Restaurant_longitude, \
                                                           row.Delivery_location_latitude,row.Delivery_location_longitude),axis=1)
            
            
        
            target_column_name = 'Time_taken (min)'
            drop_columns = [target_column_name,'ID', 'Delivery_person_ID','Month','Year']


            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]

            # logging.info(f"Modified dat before feading it to transformation pipeline: \n{input_feature_train_df.head().to_string()}")
            input_feature_train_df.head().to_csv('mod.csv')

            
            numerical_cols = input_feature_train_df.columns[input_feature_train_df.dtypes!='object']
            categorical_cols = input_feature_train_df.columns[input_feature_train_df.dtypes=='object']
            

            logging.info('Obtaining preprocessing object')
            preprocessing_obj = self.get_data_transformation_object(numerical_cols,categorical_cols)
            

            ## Trnasformating using preprocessor obj
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")
            

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys) 