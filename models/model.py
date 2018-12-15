
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

    def eval(self, next_tag, word_num, previous_tags, sentence):
        """

        :param next_tag: Next tag
        :type next_tag: int
        :param word_num: Number of word in the sentence
        :type word_num: int
        :param previous_tags: [t_-2, t_-1] first index list of -2 position tag, second tag for -1 position tag
        :type previous_tags: [list, int]
        :param sentence: List of words
        :type sentence: List of strings
        :return: List of Probabilities of next_tag
        :rtype: List of Floats
        """


    def _vectorize(self):

        vectors = []
        for i in range(self.x.shape[0]):
            a= Ratnaparkhi(self.x.loc[i,:],self.y.loc[i,:],tests=self.tests)
            vectors.append(a.fill_test())
        self.vector_x_y = np.array(vectors,dtype=Ratnaparkhi)
        return vectors

    def _loss(self, v):
        positive = self._calculate_positive(v)
        normalization = self._calculate_nonlinear(v)
        penalti = 0.5 * np.linalg.norm(v)

        return positive + normalization + penalti

    def _calculate_positive(self, v):

        matrix = []
        for mat in self.vector_x_y:
            assert isinstance(mat,Ratnaparkhi)
            matrix.append(mat.f_x_y)
        matrix.x_y = np.hstack(matrix)


        pass

    def _calculate_nonlinear(self, v):
        pass