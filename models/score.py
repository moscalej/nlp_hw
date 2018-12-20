import numpy as np
import pandas as pd


class Score:
    def __init__(self, tags):
        assert isinstance(tags, pd.Series)
        self.tags = tags
        self.cf = pd.DataFrame(index=tags, columns=tags)  # confusion Matrix
        self.x = pd.Series()
        self.y = pd.Series()

    def fit(self, y, y_hat):
        assert isinstance(y, pd.Series)
        assert isinstance(y_hat, pd.Series)
        self.y = y[y != '<PAD>'][y != '<STOP>'][y != '*']
        self.y_hat = y_hat[y_hat != '<PAD>'][y_hat != '<STOP>'][y_hat != '*']

    def matrix_confusion(self):
        y = self.y
        y_hat = self.y_hat
        most_reacuent_tags = pd.Series(self.tags.values, index=self.tags)

        for tag in most_reacuent_tags:
            places = y[y == tag].index
            fs = y_hat[places].value_counts()
            temp = most_reacuent_tags.map(fs)
            self.cf[tag] = temp
        self.cf.fillna(0, inplace=True)
        score = np.trace(self.cf) / np.sum(np.sum(self.cf))
        return score

    def over_all_acc(self):
        return np.mean(self.y == self.y_hat)

    def most_freq_tags_acc(self):
        y = self.y
        y_hat = self.y_hat
