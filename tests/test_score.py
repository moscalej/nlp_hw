import unittest
from models.score import Score
import pandas as pd
import numpy as np
from models.prerocesing import PreprocessTags

x = pd.Series(['*', '*', 'The', 'Treasury', 'is', 'still', 'working', 'out',
               'the', 'details', 'with', 'bank', 'trade', 'associations',
               'and', 'the', 'other', 'government', 'agencies',
               'that', 'have', 'a', 'hand', 'in', 'fighting', "preencounte", 'word', '<STOP>'])
y = pd.Series(['*', '*',
               'DT', 'NNP', 'VBZ', 'RB', 'VBG', 'RP',
               'DT', 'NNS', 'IN', 'NN', 'NN', 'NNS',
               'CC', 'DT', 'JJ', 'NN', 'NNS', 'WDT',
               'VBP', 'DT', 'NN', 'IN', 'VBG', 'NN', 'Vt',
               '<STOP>'])

y_bad = pd.Series(['*', '*',
               'DT', 'NNP', 'VBZ', 'RB', 'VBG', 'RP',
               'RP', 'NNS', 'IN', 'NN', 'NN', 'NNS',
               'RP', 'DT', 'NN', 'NN', 'NNS', 'WDT',
               'VBP', 'DT', 'NN', 'IN', 'IN', 'NN', 'IN',
               '<STOP>'])
tags= pd.Series(y.unique()[2:-2])

class test_Scode(unittest.TestCase):
    def test_create(self):
        sc = Score(tags)

    def test_fit(self):
        sc = Score(tags)
        sc.fit(y,y)
        self.assertTrue(np.min(sc.y_hat == y[2:-1]),'The parsing is not working')

    def test_matrix_confusion(self):
        sc = Score(tags)
        sc.fit(y, y)
        cm = sc.matrix_confusion()
        print(cm)
        sc = Score(tags)
        sc.fit(y, y_bad)
        cm2 = sc.matrix_confusion()
        print(cm2)
        eq = np.min(np.min(cm ==cm2))
        self.assertEqual(eq,0)
        same_amount = np.min(cm.sum() == cm2.sum())
        self.assertEqual(same_amount, 1)


    def test_over_all_acc(self):
        sc = Score(tags)
        sc.fit(y, y)
        cm = sc.over_all_acc()
        self.assertEqual(cm,1.)
        print(cm)

        sc = Score(tags)
        sc.fit(y, y_bad)
        cm = sc.over_all_acc()
        self.assertNotEqual(cm, 1.)
        print(cm)

    def test_all_data(self):
        data = PreprocessTags().load_data(
            r'..\data\test.wtag')
        y_hat = data.y
        y = data.y
        roll_y = pd.Series(y.values.reshape(-1))
        roll_y_hat = pd.Series(y_hat.values.reshape(-1))
        index = pd.value_counts(y.values.reshape(-1)).index
        most_reacuent_tags = pd.Series(index, index=index).drop(['<STOP>','*'])[:10]
        sc = Score(most_reacuent_tags)
        sc.fit(roll_y, roll_y_hat)
        cm = sc.matrix_confusion()
        print(cm)
        # sc = Score(tags)
        # sc.fit(y, y_bad)
        # cm2 = sc.matrix_confusion()
        # print(cm2)
        # eq = np.min(np.min(cm == cm2))
        # self.assertEqual(eq, 0)
        # same_amount = np.min(cm.sum() == cm2.sum())
        # self.assertEqual(same_amount, 1)


if __name__ == '__main__':
    unittest.main()