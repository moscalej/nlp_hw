import pandas as pd
import numpy as np


class Ratnaparkhi:
    def __init__(self, x, y):
        assert isinstance(x, pd.Series)
        assert isinstance(y, pd.Series)
        self.x = x
        self.y = y

    def f_100(self, place):
        return 1 if self.x[place] == 'base' and self.y[place] == 'Vt' else 0

    def f_101(self, place):
        if len(self.x[place]) <4 :return 0
        return 1 if self.x[place][-3:] == 'ing' and self.y[place] == 'VBG' else 0

    def f_102(self, place):
        return 1 if self.x[place][:3] == 'pre' and self.y[place] == 'NN' else 0

    def f_103(self, place):
        return 1 if np.min(self.y[[place, place - 1, place - 2]] == ['Vt', 'JJ', 'DT']) else 0

    def f_104(self, place):
        return 1 if np.min(self.y[[place, place - 1]] == ['Vt', 'JJ']) else 0

    def f_105(self, place):
        return 1 if self.y[place] == 'Vt' else 0

    def f_106(self, place):
        return 1 if self.x[place - 1] == 'the' and self.y[place] == 'Vt' else 0

    def f_107(self, place):
        return 1 if self.y[place + 1] == 'the' and self.y[place] == 'Vt' else 0
