from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainPipeline:
    def run_pipeline(self):
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()

        transformer = DataTransformation()
        train_array, test_array = transformer.initiate_data_transformation(train_path, test_path)

        trainer = ModelTrainer()
        r2 = trainer.initiate_model_trainer(train_array, test_array)
        print(f"Model trained. R2 score: {r2}")
        return r2
