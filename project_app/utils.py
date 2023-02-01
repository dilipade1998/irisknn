import numpy as np
import pandas as pd
import pickle
import json
import config

class iris_prediction():
    def __init__(self,SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm):
        self.SepalLengthCm = SepalLengthCm
        self.SepalWidthCm = SepalWidthCm
        self.PetalLengthCm = PetalLengthCm
        self.PetalWidthCm = PetalWidthCm

    def load_model(self):
        with open(config.Model_file_path, "rb") as f:
            self.Model_predict=pickle.load(f)

        with open(config.Json_file_path, "r") as f:
            self.project_json=json.load(f)

    def predict_species(self):
        self.load_model()
        test_array=np.zeros(len(self.project_json["columns"]))
        test_array[0]=self.SepalLengthCm
        test_array[1]=self.SepalWidthCm
        test_array[2]=self.PetalLengthCm
        test_array[3]=self.PetalWidthCm

        prediction=self.Model_predict.predict([test_array])
        return prediction