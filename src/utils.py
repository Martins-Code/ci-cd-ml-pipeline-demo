import os, sys
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score
from src.logger import logging

def save_object(file_path, obj):
    """
    Save any Python object to a file using dill.

    Args:
        file_path (str): Path to save the object.
        obj: The object to save.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file:
            dill.dump(obj, file)
    except Exception as e:
        raise CustomException(e)


def evaluate_model(X_train, y_train, X_test, y_test, models):
    """
    Train models and evaluate performance on training and testing datasets.

    Args:
        X_train (array): Training features.
        y_train (array): Training target values.
        X_test (array): Testing features.
        y_test (array): Testing target values.
        models (dict): Dictionary of model instances keyed by their names.

    Returns:
        dict: A report containing training and testing R^2 scores for each model.
    """
    report = {}
    try:
        for model_name, model in models.items():
            try:
                logging.info(f"Training {model_name}...")
                model.fit(X_train, y_train)

                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                train_score = r2_score(y_train, y_train_pred)
                test_score = r2_score(y_test, y_test_pred)

                logging.info(f"{model_name} - Train Score: {train_score}, Test Score: {test_score}")

                report[model_name] = {
                    'train_score': train_score,
                    'test_score': test_score
                }
            except Exception as e:
                logging.error(f"Error evaluating model {model_name}: {e}")
                report[model_name] = {'error': str(e)}

        return report
    except Exception as e:
        raise CustomException(e, sys)