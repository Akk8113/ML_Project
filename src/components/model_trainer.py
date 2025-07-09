import os
import sys
from dataclasses import dataclass
import time

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifact", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            start_time = time.time()
            logging.info("Split training and test input data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            # ✅ Get both the R2 scores and the trained model objects
            model_report, trained_models = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )

            print(f"Model report: {model_report}")

            if not model_report:
                raise CustomException("No models were successfully trained and evaluated.", sys)

            try:
                # ✅ Identify the best model based on the highest R2 score
                best_model_score = max(sorted(model_report.values()))
                best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
                
                # ✅ Get the best trained (fitted) model
                best_model = trained_models[best_model_name]

                if best_model_score < 0.6:
                    raise CustomException("No best model found", sys)

                logging.info(f"Best found model on both training and testing dataset: {best_model_name}")

                # ✅ Save the trained model
                save_object(
                    file_path=self.model_trainer_config.trained_model_file_path,
                    obj=best_model
                )

                print(f"Model saved to: {self.model_trainer_config.trained_model_file_path}")

                # ✅ Predict using the trained model
                predicted = best_model.predict(X_test)
                r2_square = r2_score(y_test, predicted)

                end_time = time.time()
                print(f"R2 score: {r2_square}")
                print(f"Total training time: {end_time - start_time} seconds")
                return r2_square

            except Exception as e:
                raise CustomException(f"Error selecting or saving best model: {e}", sys)

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        from src.components.data_ingestion import DataIngestion
        from src.components.data_transformation import DataTransformation

        data_ingestion = DataIngestion()
        train_path, test_path = data_ingestion.initiate_data_ingestion()

        data_transformation = DataTransformation()
        train_array, test_array = data_transformation.initiate_data_transformation(train_path, test_path)

        model_trainer = ModelTrainer()
        r2_score = model_trainer.initiate_model_trainer(train_array, test_array)
        print(f"R2 score of the best model: {r2_score}")

    except Exception as e:
        print(e)
