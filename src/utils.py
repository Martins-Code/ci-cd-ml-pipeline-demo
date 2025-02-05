import os, sys
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score
from src.logger import logging
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

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


def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    """
    Train models with hyperparameter tuning using RandomizedSearchCV and evaluate performance on training and testing datasets.

    Args:
        X_train (array): Training features.
        y_train (array): Training target values.
        X_test (array): Testing features.
        y_test (array): Testing target values.
        models (dict): Dictionary of model instances keyed by their names.
        params (dict): Dictionary of hyperparameter grids for each model.

    Returns:
        dict: A report containing training and testing R^2 scores, along with best parameters for each model.
    """
    report = {}
    try:
        for model_name, model in models.items():
            try:
                logging.info(f"Training {model_name} with RandomizedSearchCV...")

                # Get hyperparameter grid for the current model, if available.
                param_grid = params.get(model_name, {})

                if param_grid:
                    # Use RandomizedSearchCV to randomly sample a fixed number of parameter settings
                    random_search = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=param_grid,
                        cv=3,                      # Reduced number of CV folds to 3
                        scoring="r2",
                        n_iter=10,                 # Try only 10 combinations to reduce the search space
                        n_jobs=-1,
                        verbose=1,
                        random_state=42
                    )
                    random_search.fit(X_train, y_train)
                    best_model = random_search.best_estimator_
                    best_params = random_search.best_params_
                    logging.info(f"Best parameters for {model_name}: {best_params}")
                else:
                    best_model = model
                    best_model.fit(X_train, y_train)
                    best_params = "Default parameters"

                # Make predictions
                y_train_pred = best_model.predict(X_train)
                y_test_pred = best_model.predict(X_test)

                # Evaluate model performance using R^2 score
                train_score = r2_score(y_train, y_train_pred)
                test_score = r2_score(y_test, y_test_pred)

                logging.info(f"{model_name} - Train Score: {train_score:.4f}, Test Score: {test_score:.4f}")

                # Store the results
                report[model_name] = {
                    'train_score': train_score,
                    'test_score': test_score,
                    'best_params': best_params
                }
            except Exception as e:
                logging.error(f"Error evaluating model {model_name}: {e}")
                report[model_name] = {'error': str(e)}

        return report
    except Exception as e:
        raise CustomException(e, sys)
