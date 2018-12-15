
import pandas as pd
import numpy as np
from models.features import Ratnaparkhi


class Model:
    def __init__(self, tests):
        self.tests = tests
        self.v = None
        self.x = None
        self.y = None
        self.vector_x_y=None
        self.y_values = None

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
        self.x = x
        self.y = y
        self.y_values = pd.unique(y.values.ravel('K'))
        self._vectorize()

        return

    def predict(self,x):
        """
        This will work with the Viterbi
        :param x:
        :return:
        """
        pass

    def _vectorize(self):

        vectors = []
        for i in range(self.x.shape[0]):
            a= Ratnaparkhi(self.x.loc[i,:],self.y.loc[i,:],tests=self.tests,y_corpus=self.y_values)
            vectors.append(a.fill_test())
        self.vector_x_y = np.array(vectors,dtype=Ratnaparkhi)


    def _loss(self, v):
        positive = self._calculate_positive(v)
        normalization = self._calculate_nonlinear(v)
        penalti = 0.5 * np.linalg.norm(v)

        return positive + normalization + penalti

    def _calculate_positive(self, v):
        """
        This method will solve the positive part of the loss Function
        for all the sentence
        sum (sum (v * f(h_i^(k),y_i), for i=0 to max sise word), for k =0 to last sentence)
        = sum( F dot v ) where F is concatenate matrix for all the vectors f
        :param v:
        :return:
        """
        assert isinstance(v, np.ndarray)
        matrix = []
        for mat in self.vector_x_y:
            assert isinstance(mat,Ratnaparkhi)
            matrix.append(mat.f_x_y)
        matrix_x_y = np.concatenate(matrix, axis=0)
        dot_m = matrix_x_y @ v
        return dot_m.sum()

    def _calculate_nonlinear(self, v):
        assert isinstance(v, np.ndarray)
        matrix = []
        for mat in self.vector_x_y:
            assert isinstance(mat, Ratnaparkhi)
            matrix.append(mat.non_lineard_sentence(v))
        return np.sum(matrix)