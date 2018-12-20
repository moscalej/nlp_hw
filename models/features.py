import pandas as pd
import numpy as np


class FinkMos:

    def __init__(self, x, y, tests, tag_corpus):
        assert isinstance(x, pd.Series)
        assert isinstance(y, pd.Series)
        self.tag_corpus = tag_corpus
        self.tests = tests
        self.x = x
        self.y = y
        self.f_matrix = np.empty(self.y.shape,dtype=np.ndarray)
        self.f_x_y = pd.DataFrame(np.zeros([self.x.shape[0], len(tests)]), columns=tests)  # TODO change this sise

    def f_100(self, place, y, y_1, y_2):
        return 1 if self.x[place] == 'base' and y == 'Vt' else 0

    def f_101(self, place, y, y_1, y_2):
        if len(self.x[place]) < 4:
            return 0
        return 1 if self.x[place][-3:] == 'ing' and y == 'VBG' else 0

    def f_102(self, place, y, y_1, y_2):
        if len(self.x[place]) < 4:
            return 0
        return 1 if self.x[place][:3] == 'pre' and y == 'NN' else 0

    def f_103(self, place, y, y_1, y_2):
        return 1 if y == 'Vt' and y_1 == 'JJ' and y_2 == 'DT' else 0

    def f_104(self, place, y, y_1, y_2):
        return 1 if y == 'Vt' and y_1 == 'JJ' else 0

    def f_105(self, place, y, y_1, y_2):
        return 1 if y == 'Vt' else 0

    def f_106(self, place, y, y_1, y_2):
        return 1 if self.x[place - 1] == 'the' and y == 'Vt' else 0

    def f_107(self, place, y, y_1, y_2):
        if place == self.x.size - 1:
            return 0
        return 1 if self.x[place + 1] == 'the' and y == 'Vt' else 0

    def fill_test(self):
        """

        :param tests:
        :return: Return vector where each value is the value of a test
        """
        for test in self.tests:
            self.run_line_tests(test)
        return self

    def run_line_tests(self, test_name):

        test = getattr(self, test_name)
        list_1 = [0, 0, 0]
        for i in range(3, self.x.size):
            list_1.append(test(i, self.y[i], self.y[i-1], self.y[i-2]))
        self.f_x_y.loc[:, test_name] = list_1

    def to_feature_space(self, history_i, y):
        results = []
        for test_ in self.tests:
            test = getattr(self, test_)
            results.append(test(history_i, y, self.y[history_i-1], self.y[history_i-2]))
        return np.array([results])  # todo check if np.array is faster or list


    def to_feature_space2(self, history_i, y, y_1, y_2):
        results = []
        for test in self.tests:
            test = getattr(self, test)
            results.append(test(history_i, y, y_1, y_2))
        return np.array([results])  # todo check if np.array is faster or list


    def sentence_non_linear_loss_inner2(self, v, history_i, y, y_2):
        if self.f_matrix is None:
            results = []
            for tag in self.tag_corpus:
                results.append(self.to_feature_space2(history_i, y, tag, y_2))
            f_matrix = np.concatenate(results, axis=0)
            self.f_matrix = f_matrix

        v_f = self.f_matrix @ v
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
        end = self.y[self.y == '<STOP>'].index[0]
        values = []
        for word in range(2, end):
            values.append(self.sentence_non_linear_loss_inner(v, word))
        sentence_normal = np.sum(np.log(np.array(values)))
        return sentence_normal




class CustomFeatures:
    def __init__(self, x, y):
        assert isinstance(x, pd.Series)
        assert isinstance(y, pd.Series)
        self.x = x
        self.y = y
