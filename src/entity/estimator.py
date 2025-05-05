import sys

import pandas as pd
from pandas import DataFrame
from sklearn.pipeline import Pipeline

from src.exception import MyException
from src.logger import logging

class TargetValueMapping:
    def __init__(self):
        # self.yes:int = 0
        # self.no:int = 1
    
           self.Apple:int=0
           self.Banana:int = 1
           self.Blackgram:int = 2
           self.ChickPea:int = 3
           self.Coconut:int = 4
           self.Coffee:int = 5
           self.Cotton:int = 6
           self.Grapes:int = 7
           self.Jute:int = 8
           self.Kidneybeans:int = 9
           self.Lentil:int = 10
           self.Maize:int = 11
           self.Mango:int = 12
           self.MothBeans:int = 13
           self.MungBean:int = 14
           self.Muskmelon:int = 15
           self.Orange:int = 16
           self.Papaya:int = 17
           self.PigeonPeas:int = 18
           self.Pomegranate:int = 19
           self.Rice:int = 20
           self.Watermelon:int = 21
    def _asdict(self):
        return self.__dict__
    def reverse_mapping(self):
        mapping_response = self._asdict()
        return dict(zip(mapping_response.values(),mapping_response.keys()))

class MyModel:
    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
        """
        :param preprocessing_object: Input Object of preprocesser
        :param trained_model_object: Input Object of trained model 
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, dataframe: pd.DataFrame) -> DataFrame:
        """
        Function accepts preprocessed inputs (with all custom transformations already applied),
        applies scaling using preprocessing_object, and performs prediction on transformed features.
        """
        try:
            logging.info("Starting prediction process.")

            # Step 1: Apply scaling transformations using the pre-trained preprocessing object
            transformed_feature = self.preprocessing_object.transform(dataframe)

            # Step 2: Perform prediction using the trained model
            logging.info("Using the trained model to get predictions")
            predictions = self.trained_model_object.predict(transformed_feature)

            return predictions

        except Exception as e:
            logging.error("Error occurred in predict method", exc_info=True)
            raise MyException(e, sys) from e


    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"