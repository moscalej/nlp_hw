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

    def f_100(self, place):
        return 1 if self.x[place] == 'base' and self.y[place] == 'Vt' else 0

    def f_101(self, place):
        if len(self.x[place]) < 4:
            return 0
        return 1 if self.x[place][-3:] == 'ing' and self.y[place] == 'VBG' else 0

    def f_102(self, place):
        if len(self.x[place]) < 4:
            return 0
        return 1 if self.x[place][:3] == 'pre' and self.y[place] == 'NN' else 0

    def f_103(self, place):
        return 1 if self.y[place] == 'Vt' and self.y[place - 1] == 'JJ' and self.y[place - 2] == 'DT' else 0

    def f_104(self, place):
        return 1 if self.y[place] == 'Vt' and self.y[place - 1] == 'JJ' else 0

    def f_105(self, place):
        return 1 if self.y[place] == 'Vt' else 0

    def f_106(self, place):
        return 1 if self.x[place - 1] == 'the' and self.y[place] == 'Vt' else 0

    def f_107(self, place):
        if place == self.y.size - 1:
            return 0
        return 1 if self.y[place + 1] == 'the' and self.y[place] == 'Vt' else 0

    def fill_test(self):
        """

        :param tests:
        :return: Return vector where each value is the value of a test
        """
        for test in self.tests:
            self.run_line_tests(test)
        return self.f_x_y

    def run_line_tests(self, test_name):

        test = getattr(self, test_name)
        list_1 =[0,0,0]
        for i in range(3, self.x.size):
            list_1.append(test(i))
        self.f_x_y.loc[:, test_name] = list_1


class CustomFeatures:
    def __init__(self, x, y):
        assert isinstance(x, pd.Series)
        assert isinstance(y, pd.Series)
        self.x = x
        self.y = y
