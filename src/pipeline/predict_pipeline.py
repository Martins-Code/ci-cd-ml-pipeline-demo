import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    """
    A class to handle the machine learning prediction pipeline.
    """

    def __init__(self):
        pass

    def predict(self, input_features):
        """
        Predicts the output for given input features.

        Parameters:
        - input_features (pd.DataFrame): Preprocessed input data.

        Returns:
        - np.ndarray: Model predictions.
        """
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'

            # Load model and preprocessor
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            # Apply preprocessing and make prediction
            scaled_data = preprocessor.transform(input_features)
            predictions = model.predict(scaled_data)

            return predictions

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    A class to structure and process user-provided input data for model prediction.
    """

    def __init__(self, gender, race_ethnicity, parental_level_of_education,
                 lunch, test_preparation_course, reading_score, writing_score):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def to_dataframe(self):
        """
        Converts the collected user data into a Pandas DataFrame for model input.

        Returns:
        - pd.DataFrame: The structured data.
        """
        try:
            student_data = {
                'gender': [self.gender],
                'race_ethnicity': [self.race_ethnicity],
                'parental_level_of_education': [self.parental_level_of_education],
                'lunch': [self.lunch],
                'test_preparation_course': [self.test_preparation_course],
                'reading_score': [self.reading_score],
                'writing_score': [self.writing_score]
            }
            return pd.DataFrame(student_data)

        except Exception as e:
            raise CustomException(e, sys)
