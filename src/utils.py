import os
import sys
import pickle
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException

def save_object(file_path: str, obj: object) -> None:
    """
    Save a Python object to a file using pickle.
    Creates the directory if it does not exist.

    Args:
        file_path (str): The file path where the object will be saved.
        obj (object): The Python object to save.

    Raises:
        CustomException: If there is an error during saving.
    """
    try:
        dir_path = os.path.dirname(file_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models: dict, param: dict) -> dict:
    """
    Train and evaluate multiple models with default parameters (no hyperparameter tuning).

    Args:
        X_train: Training features.
        y_train: Training target.
        X_test: Testing features.
        y_test: Testing target.
        models (dict): Dictionary of model name to model instance.
        param (dict): Ignored in this simplified version.

    Returns:
        dict: Model name to R2 score on test data.
    """
    from sklearn.metrics import r2_score
    model_report = {}

    for model_name, model in models.items():
        try:
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            model_report[model_name] = test_score

        except Exception as e:
            logging.error(f"Model {model_name} failed during training or evaluation: {e}")

    return model_report
