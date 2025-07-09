import pandas as pd
import os
import pickle

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifact", "model.pkl")
        self.preprocessor_path = os.path.join("artifact", "preprocessor.pkl")

    def predict(self, features: pd.DataFrame):
        preprocessor = pickle.load(open(self.preprocessor_path, 'rb'))
        model = pickle.load(open(self.model_path, 'rb'))
        data_transformed = preprocessor.transform(features)
        predictions = model.predict(data_transformed)
        return predictions