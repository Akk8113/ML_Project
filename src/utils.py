import os
import sys
import pickle

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
