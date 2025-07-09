import sys
import os
import pandas as pd
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from sklearn.compose import ColumnTransformer


@dataclass
class DataTransformationConfig:
    artifact_dir: str = os.path.join('artifact')
    preprocessor_obj_file_path: str = os.path.join(artifact_dir, 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, numerical_features, categorical_features):
        try:
            logging.info("Creating data transformation pipelines for numerical and categorical features")

            # Numerical pipeline
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            # Categorical pipeline
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore')),
                ('scaler', StandardScaler(with_mean=False))
            ])
            
            logging.info(f"Numercal columns standard scaling completed.{numerical_features}")
            logging.info(f"Categorical columns encoding completed.{categorical_features}")

            # Combine pipelines
            preprocessor = ColumnTransformer(transformers=[
                ('num', num_pipeline, numerical_features),
                ('cat', cat_pipeline, categorical_features)
            ])

            logging.info("Data transformation pipelines created successfully")
            return preprocessor

        except Exception as e:
            logging.error("Error in creating data transformation pipelines")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Starting data transformation process")

            # Read data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Identify numerical and categorical features
            target_column = 'Good/Bad'  # Updated to actual target column name from dataset
            numerical_features = train_df.drop(columns=[target_column]).select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_features = train_df.drop(columns=[target_column]).select_dtypes(include=['object', 'category']).columns.tolist()

            logging.info(f"Numerical features: {numerical_features}")
            logging.info(f"Categorical features: {categorical_features}")

            # Get transformer object
            preprocessor = self.get_data_transformer_object(numerical_features, categorical_features)

            # Separate features and target
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            # Fit and transform train data
            X_train_transformed = preprocessor.fit_transform(X_train)

            # Transform test data
            X_test_transformed = preprocessor.transform(X_test)

            # Concatenate transformed features with target to form final arrays
            import numpy as np
            train_array = np.c_[X_train_transformed, y_train.to_numpy()]
            test_array = np.c_[X_test_transformed, y_test.to_numpy()]

            # Save the preprocessor object using utils.save_object
            save_object(self.data_transformation_config.preprocessor_obj_file_path, preprocessor)

            logging.info("Data transformation process completed successfully")

            return train_array, test_array

        except Exception as e:
            logging.error("Error in data transformation process")
            raise CustomException(e, sys)
