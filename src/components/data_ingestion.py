import os 
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig


@dataclass ## Decorator.
class DataIngestionConfig:
    artifact_dir : str = 'artifact'
    train_data_path: str = os.path.join('artifact', "train.csv")
    test_data_path: str = os.path.join('artifact', "test.csv")
    raw_data_path: str = os.path.join('artifact', "raw.csv")

    
class DataIngestion:
    def __init__(self): ## constructor.
        self.ingestion_config= DataIngestionConfig()

    def initiate_data_ingestion(self):
        print("--- initiate_data_ingestion called with the LATEST CODE ---")
        logging.info("Enter the data ingestion method or component")
        try:
            # Construct the absolute path to your data file
            # Assuming 'notebook/data/wafer_data.csv' is relative to D:\Water_Fault_Project
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            print(f"Current file directory: {current_file_dir}")
            # Go up two levels to get to the project root (from src/components to Water_Fault_Project)
            project_root = os.path.abspath(os.path.join(current_file_dir, '..', '..'))
            print(f"Project root directory: {project_root}")
            data_file_path = os.path.join(project_root, 'notebook', 'data', 'wafer_data.csv')
            print(f"Data file path: {data_file_path}")
            
            df = pd.read_csv(data_file_path)
            print("Dataset read successfully")
            logging.info("Read the dataset as dataframe.")

            # Create the 'artifact' directory if it doesn't exist
            # Use os.makedirs with the directory path for 'artifact'
            artifact_full_path = os.path.join(project_root, self.ingestion_config.artifact_dir)
            print(f"Artifact directory path: {artifact_full_path}")
            os.makedirs(artifact_full_path, exist_ok=True)
            print(f"Artifact directory created or already exists")
            logging.info(f"Ensured 'artifact' directory exists at: {artifact_full_path}")

            # Save the raw data
            raw_data_full_path = os.path.join(project_root, self.ingestion_config.raw_data_path)
            print(f"Saving raw data to: {raw_data_full_path}")
            df.to_csv(raw_data_full_path, index=False, header=True)
            print("Raw data saved successfully")
            logging.info(f"Raw data saved to: {raw_data_full_path}")

            logging.info("Train Test split initiated ")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save the train set
            train_data_full_path = os.path.join(project_root, self.ingestion_config.train_data_path)
            print(f"Saving train data to: {train_data_full_path}")
            train_set.to_csv(train_data_full_path, index=False, header=True)
            print("Train data saved successfully")
            logging.info(f"Train data saved to: {train_data_full_path}")

            # Save the test set
            test_data_full_path = os.path.join(project_root, self.ingestion_config.test_data_path)
            print(f"Saving test data to: {test_data_full_path}")
            test_set.to_csv(test_data_full_path, index=False, header=True)
            print("Test data saved successfully")
            logging.info(f"Test data saved to: {test_data_full_path}")

            logging.info("Ingestion of the data is completed")

            return (
                train_data_full_path,
                test_data_full_path
            )
        except Exception as e:
            print(f"Exception occurred: {e}")
            raise CustomException(e, sys)
            
if __name__=='__main__':
    obj= DataIngestion()
    train_path, test_path =obj.initiate_data_ingestion()
    print(f"Train data available at: {train_path}")
    print(f"Test data available at: {test_path}")

    datatransformation = DataTransformation()
    train_array, test_array = datatransformation.initiate_data_transformation(train_path,test_path)

    from src.components.model_trainer import ModelTrainer

    model_trainer = ModelTrainer()
    r2_score = model_trainer.initiate_model_trainer(train_array, test_array)
    print(f"R2 score of the best model: {r2_score}")
    