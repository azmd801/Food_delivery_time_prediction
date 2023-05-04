import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
import numpy as np
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


def main() -> None:
    # Initiate Data Ingestion
    data_ingestion = DataIngestion()
    train_data_path: str
    test_data_path: str
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

    # Initiate Data Transformation
    data_transformation = DataTransformation()
    train_arr: np.array
    test_arr: np.array
    train_arr, test_arr,_ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

    # Initiate Model Training
    model_trainer = ModelTrainer()
    model_trainer.initate_model_training(train_arr, test_arr)


if __name__ == '__main__':
    main()