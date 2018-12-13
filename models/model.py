
import pandas as pd
import numpy as np
from models.features import Ratnaparkhi


class Model:
    def __init__(self, tests=[]):
        self.test = tests
        self.v = None
        self.x = None
        self.y = None
        self.vector_x=None

    def fit(self,x , y, learning_rate=0.02,x_val=None , y_val = None):
        """
        Fit will train the Model
            - Encoding
                - For the data will create a Tensor [ # todo Read more
            - Gradient decent
                -Loss
                - update V
            - Calculate metrics

        :param x: DataFrame [row = sentence , col = word]
        :param y: DataFrame [row = sentence tag , col = Word tag]
        :param learning_rate: [don't know if need it] # TODO check if remove
        :param x_val:[row = sentence , col = word]
        :param y_val:[row = sentence tag , col = Word tag]
        :return: metrics dict {} $# TODO check witch metrics we need
        """
        assert isinstance(x,pd.DataFrame)
        assert isinstance(y, pd.DataFrame)
        self._vectorize(x,y)

        pass

    def predict(self,x):
        """
        This will work with the Viterbi
        :param x:
        :return:
        """
        pass

    def _vectorize(self, x, y):

        self.vector_x = pd.DataFrame([
            Ratnaparkhi(x_row[1], y_row[1])
                .fill_test(self.test) for x_row, y_row in zip(x.iterrows(), y.iterrows())
        ])
        return self.vector_x
