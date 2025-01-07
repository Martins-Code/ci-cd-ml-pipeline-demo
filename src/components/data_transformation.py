import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass(frozen=True)
class DataTransformationConfig:
    """
    Configuration for saving the preprocessor object.
    """
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_preprocessing_pipeline(self):
        """
        Creates pipelines for preprocessing numerical and categorical data.

        Returns:
            ColumnTransformer: A complete preprocessing pipeline.
        """
        try:
            numeric_features = ['reading_score', 'writing_score']
            categorical_features = [
                'gender', 'race_ethnicity', 'parental_level_of_education', 
                'lunch', 'test_preparation_course'
            ]

            numeric_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy="median")), 
                ('scaler', StandardScaler())
            ])

            categorical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False))
            ])

            logging.info("Preprocessing pipelines created successfully.")

            return ColumnTransformer([
                ('numeric_pipeline', numeric_pipeline, numeric_features),
                ('categorical_pipeline', categorical_pipeline, categorical_features)
            ])

        except Exception as e:
            raise CustomException(e, sys)

    def initialize_data_transformation(self, train_data_path: str, test_data_path: str):
        """
        Transforms training and testing data, saves preprocessing pipeline.

        Returns:
            tuple: Transformed train/test data and saved preprocessor path.
        """
        try:
            # Load datasets
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            logging.info("Training and testing data loaded.")

            target_column = 'math_score'
            preprocessor = self.get_preprocessing_pipeline()

            # Separate features and targets
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]
            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            # Preprocess data
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            # Combine features and target
            train_data = np.c_[X_train_transformed, y_train.values]
            test_data = np.c_[X_test_transformed, y_test.values]
            logging.info("Data transformation completed successfully.")

            # Save the preprocessor object
            save_object(self.config.preprocessor_obj_file_path, preprocessor)
            logging.info("Preprocessing object saved.")

            return train_data, test_data, self.config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
