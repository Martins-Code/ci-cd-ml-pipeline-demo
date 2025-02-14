import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRFRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model


@dataclass(frozen=True)
class ModelTrainerConfig:
    """
    Configuration for saving the model object.
    """
    model_obj_file_path: str = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        # Initialize configuration for saving the model object
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """
        Trains multiple models, evaluates them, and saves the best-performing model.
        
        :param train_array: Array containing training features and target variable.
        :param test_array: Array containing testing features and target variable.
        """
        try:
            logging.info("Splitting training and test input data.")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            # Define models to train
            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "KNN": KNeighborsRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "XGBoost": XGBRFRegressor(),
            }

            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter': ['best', 'random'],  # Controls how splits are chosen
                    'max_features': ['sqrt', 'log2'],  # Limits features per split
                },
                "Random Forest": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_features': ['sqrt', 'log2', None],  # Feature selection strategies
                    'n_estimators': [50, 100, 200, 300],  # Number of trees in the forest
                },
                "Gradient Boosting": {
                    'loss': ['squared_error', 'huber', 'absolute_error', 'quantile'],  # Loss function selection
                    'learning_rate': [0.1, 0.05, 0.01, 0.001],  # Step size
                    'subsample': [0.6, 0.75, 0.85, 0.95],  # Fraction of samples per boosting round
                    'criterion': ['squared_error', 'friedman_mse'],
                    'max_features': ['auto', 'sqrt', 'log2'],  # Feature selection strategies
                    'n_estimators': [50, 100, 200, 300],  # Number of boosting stages
                },
                "Linear Regression": {},  # No hyperparameters to tune
                "XGBRegressor": {
                    'learning_rate': [0.1, 0.05, 0.01, 0.001],  # Learning rate for boosting
                    'n_estimators': [50, 100, 200, 300],  # Number of trees
                    'max_depth': [3, 5, 7],  # Tree depth for controlling complexity
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.1, 0.05, 0.01, 0.001],  # Step size shrinkage
                    'loss': ['linear', 'square', 'exponential'],  # Error weighting strategy
                    'n_estimators': [50, 100, 200, 300],  # Number of weak learners
                }
            }

            



            logging.info("Evaluating models using training and test data...")
            model_report = evaluate_model(X_train, y_train, X_test, y_test, models, params)
            logging.info(f"Model evaluation completed. Report: {model_report}")

            # Find and log the best-performing model
            try:
                logging.info("Finding the best model from the report.")

                # Extract valid test scores
                valid_scores = {
                    model_name: score['test_score']
                    for model_name, score in model_report.items()
                    if isinstance(score, dict) and 'test_score' in score
                }

                # Ensure there are valid scores to evaluate
                if not valid_scores:
                    logging.warning("No valid scores found. Cannot select the best model.")
                    raise ValueError("No valid model scores to evaluate.")

                # Get the best model score and name
                best_model_score = max(valid_scores.values())
                best_model_name = max(valid_scores, key=valid_scores.get)

                logging.info(f"Best model: {best_model_name} with a score of {best_model_score:.4f}")

                # Save the best-performing model
                logging.info("Saving the best model...")
                save_object(
                    self.model_trainer_config.model_obj_file_path,
                    models[best_model_name],
                )
                logging.info(f"Best model saved to {self.model_trainer_config.model_obj_file_path}")

            except Exception as e:
                logging.error(f"Error during best model selection: {e}")
                raise CustomException(e, sys)

        except Exception as e:
            logging.error(f"Error occurred during model training initiation: {e}")
            raise CustomException(e, sys)
