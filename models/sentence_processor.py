import pandas as pd
import numpy as np
from models.features import Features

class FinkMos:

    def __init__(self, x, y, tests, tag_corpus):
        assert isinstance(x, pd.Series)
        assert isinstance(y, pd.Series)
        self.tag_corpus = tag_corpus
        self.tests = tests
        self.test_f = Features().get_tests()
        self.x = x
        self.y = y
        self.f_matrix = np.empty(self.y.shape,dtype=np.ndarray)
        self.f_matrix_y_1 = np.empty(self.y.shape,dtype=np.ndarray)
        self.f_x_y = pd.DataFrame(np.zeros([self.x.shape[0], len(tests)]), columns=tests)  # TODO change this sise



    def fill_test(self):
        """

        :param tests:
        :return: Return vector where each value is the value of a test
        """
        for test in self.tests:
            self.run_line_tests(test)
        return self

    def run_line_tests(self, test_name):

        test = self.test_f[test_name]
        list_1 = [0, 0, 0]
        for i in range(3, self.x.size):
            list_1.append(test(self.x,i, self.y[i], self.y[i-1], self.y[i-2]))
        self.f_x_y.loc[:, test_name] = list_1

    def to_feature_space(self, history_i, y):
        results = []
        for test_ in self.tests:
            test = self.test_f[test_]
            results.append(test(self.x,history_i, y, self.y[history_i-1], self.y[history_i-2]))
        return np.array([results])  # todo check if np.array is faster or list


    def to_feature_space2(self, history_i, y, y_1, y_2):
        results = []
        for test in self.tests:
            test = self.test_f[test]
            results.append(test(self.x,history_i, y, y_1, y_2))
        return np.array([results])  # todo check if np.array is faster or list


    def sentence_non_linear_loss_inner2(self, v, history_i, y, y_2):
        results = []
        for tag in self.tag_corpus:
            results.append(self.to_feature_space2(history_i, y, tag, y_2))
        f_matrix = np.concatenate(results, axis=0)
        self.f_matrix_y_1[history_i] = f_matrix

        v_f = self.f_matrix_y_1[history_i] @ v
        e_val = np.log(np.sum(np.exp(v_f)))
        return e_val


    def sentence_non_linear_loss_inner(self, v, history_word_index):
        if self.f_matrix[history_word_index] is None:
            results = []
            for word in self.tag_corpus:
                results.append(self.to_feature_space(history_word_index, word))
            f_matrix = np.concatenate(results, axis=0)
            self.f_matrix[history_word_index] = f_matrix

        v_f = self.f_matrix[history_word_index] @ v
        e_val = np.sum(np.exp(v_f))
        return e_val


    def sentence_non_lineard_loss(self, v):
        values = []
        for word in range(2, self.y.shape[0]):
            values.append(self.sentence_non_linear_loss_inner(v, word))
        sentence_normal = np.sum(np.log(np.array(values)))
        return sentence_normal




class CustomFeatures:
    def __init__(self, x, y):
        assert isinstance(x, pd.Series)
        assert isinstance(y, pd.Series)
        self.x = x
        self.y = y
