import pandas as pd
import numpy as np


class Ratnaparkhi:

    def __init__(self, x, y, tests):
        assert isinstance(x, pd.Series)
        assert isinstance(y, pd.Series)
        self.tests = tests
        self.x = x
        self.y = y
        self.f_x_y = pd.DataFrame(np.zeros([self.x.shape[0], len(tests)]), columns=tests)  # TODO change this sise

    def f_100(self, place, y):
        return 1 if self.x[place] == 'base' and y == 'Vt' else 0

    def f_101(self, place, y):
        if len(self.x[place]) < 4:
            return 0
        return 1 if self.x[place][-3:] == 'ing' and y == 'VBG' else 0

    def f_102(self, place, y):
        if len(self.x[place]) < 4:
            return 0
        return 1 if self.x[place][:3] == 'pre' and y == 'NN' else 0

    def f_103(self, place, y):
        return 1 if y == 'Vt' and self.y[place - 1] == 'JJ' and self.y[place - 2] == 'DT' else 0

    def f_104(self, place, y):
        return 1 if y == 'Vt' and self.y[place - 1] == 'JJ' else 0

    def f_105(self, place, y):
        return 1 if y == 'Vt' else 0

    def f_106(self, place, y):
        return 1 if self.x[place - 1] == 'the' and y == 'Vt' else 0

    def f_107(self, place, y):
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
        list_1 =[0,0,0]
        for i in range(3, self.x.size):
            list_1.append(test(i, self.y[i]))
        self.f_x_y.loc[:, test_name] = list_1


class CustomFeatures:
    def __init__(self, x, y):
        assert isinstance(x, pd.Series)
        assert isinstance(y, pd.Series)
        self.x = x
        self.y = y
