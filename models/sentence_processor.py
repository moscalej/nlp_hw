import numpy as np
import pandas as pd

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
        self.f_matrix = np.empty(self.y.shape, dtype=np.ndarray)  #
        self.f_matrix_y_1 = np.empty(self.y.shape,dtype=np.ndarray)
        self.f_x_y = pd.DataFrame(np.zeros([self.x.shape[0], len(tests)]), columns=tests)
        self.word2number = {word: index for index, word in enumerate(x.value_counts().index)}
        tc = tag_corpus.shape[0]
        self.fast_test = dict()

    def fill_test(self):
        """

        :param tests:
        :return: Return vector where each value is the value of a test
        """

        for i in range(3, self.x.size):
            self.f_x_y.loc[i, :] = np.array(self.to_feature_space2(i, self.y[i], self.y[i - 1], self.y[i - 2])) * 0.1
        return self

    def sentence_non_linear_loss_inner(self, v, h_word_i):
        if self.f_matrix[h_word_i] is None:
            results = []
            for tag in self.tag_corpus:
                results.append(self.to_feature_space2(h_word_i, tag, self.y[h_word_i - 1], self.y[h_word_i - 2]))
            f_matrix = np.concatenate(results, axis=0)
            self.f_matrix[h_word_i] = f_matrix * 0.1  # A(i,j) = [i is the word in the curpus , j is the test done]
        v_f = self.f_matrix[h_word_i] @ v
        e_val = np.sum(np.exp(v_f))
        return e_val

    def sentence_non_lineard_loss(self, v):
        values = []
        for word in range(2, self.y.shape[0]):
            values.append(self.sentence_non_linear_loss_inner(v, word))
        sentence_normal = np.sum(np.log(np.array(values)))
        return sentence_normal

    def to_feature_space2(self, history_i, y, y_1, y_2):
        hash_name = f"{self.x[history_i]}{y}{y_1}{y_2}"
        if hash_name in self.fast_test:
            results = self.fast_test[hash_name]
        else:
            results = []
            for test in self.tests:
                results.append(self.test_f[test](self.x, history_i, y, y_1, y_2))
                self.fast_test[hash_name] = results
        return np.array([results])

    def softmax_denominator(self, v, history_i, y, y_1, y_2):
        results = []
        for tag in self.tag_corpus:
            results.append(self.to_feature_space2(history_i, tag, y_1, y_2))
        f_matrix = np.concatenate(results, axis=0)
        self.f_matrix_y_1[history_i] = f_matrix

        e_val = np.sum(np.exp(f_matrix @ v))
        return e_val

    def prob_q(self, v, history_i, y, y_1, y_2):
        return np.exp(self.to_feature_space2(history_i, y, y_1, y_2) @ v) / self.softmax_denominator(v, history_i, y,
                                                                                                     y_1, y_2)







class CustomFeatures:
    def __init__(self, x, y):
        assert isinstance(x, pd.Series)
        assert isinstance(y, pd.Series)
        self.x = x
        self.y = y
