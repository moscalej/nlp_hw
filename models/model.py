
import pandas as pd
import numpy as np
from models.features import FinkMos


class Model:
    def __init__(self, tests):
        self.tests = tests
        self.v = None
        self.x = None
        self.y = None
        self.vector_x_y=None
        self.tag_corpus = None

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
        self.tag_corpus = pd.unique(y.values.ravel('K')) # TODO remove '*' , '<PAD>' , '<STOP>"
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
        matrix = []
        for i in range(self.x.shape[0]):
            a= FinkMos(self.x.loc[i, :], self.y.loc[i, :], tests=self.tests, y_corpus=self.tag_corpus)
            vectors.append(a.fill_test())
            matrix.append(a.f_x_y)
        self.vector_x_y = np.array(vectors, dtype=FinkMos)

        self.lin_loss_matrix_x_y = np.concatenate(matrix, axis=0)
        # is a sentence


    def _loss(self, v):
        positive = self._calculate_positive(v)
        non_linear = self._calculate_nonlinear(v)
        penalty = 0.5 * np.linalg.norm(v)

        return positive - non_linear + penalty

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

        dot_m = self.lin_loss_matrix_x_y @ v
        return dot_m.sum()

    def _calculate_nonlinear(self, v):
        assert isinstance(v, np.ndarray)
        matrix = []
        for mat in self.vector_x_y:
            assert isinstance(mat, FinkMos)
            matrix.append(mat.sentence_non_lineard_loss(v))
        return np.sum(matrix)