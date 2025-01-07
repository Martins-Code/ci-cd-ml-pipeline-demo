import os
import dill
from src.exception import CustomException

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
