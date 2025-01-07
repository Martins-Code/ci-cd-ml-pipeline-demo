import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation, DataTransformationConfig


@dataclass
class DataIngestionConfig:
    """
    Configuration class for defining file paths during the data ingestion process.
    Attributes:
    - train_data_path: Path to save the training dataset.
    - test_data_path: Path to save the test dataset.
    - raw_data_path: Path to save the raw dataset.
    """
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')


class DataIngestion:
    """
    Class to handle data ingestion by reading raw data, splitting into train-test sets, 
    and saving the outputs to predefined locations.
    """

    def __init__(self):
        """
        Initializes the DataIngestion class by setting up configuration paths.
        """
        self.ingestion_config = DataIngestionConfig()

    def initialize_data_ingestion(self):
        """
        Handles the data ingestion process:
        - Reads raw data from a CSV file.
        - Splits the data into training and testing datasets.
        - Saves the datasets to predefined paths.
        
        Returns:
        - Tuple containing paths to the training and testing datasets.
        """
        logging.info('Entered the data ingestion method or component')
        try:
            # Load dataset
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('Dataset loaded successfully')

            # Create necessary directories
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            # Save the raw dataset
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info('Raw data saved successfully')

            # Train-test split
            logging.info('Performing train-test split')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            # Save training and test datasets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info('Train and test data saved successfully')

            # Return paths for downstream use
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            logging.error('Error occurred during data ingestion')
            raise CustomException(e, sys)


if __name__ == "__main__":
    data_inject = DataIngestion()
    train_data, test_data = data_inject.initialize_data_ingestion()

    data_transformation = DataTransformation()
    data_transformation.initialize_data_transformation(train_data, test_data)