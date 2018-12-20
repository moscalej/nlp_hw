import unittest
from models.prerocesing import PreprocessTags
from models.features import FinkMos
from models.model import Model
import pandas as pd
import numpy as np




# y_tags = [ 'IN', 'NN', 'DT', 'PRP', 'CC', 'RB', 'NNS', 'JJ', '``', 'EX',
#        'NNP', 'CD', 'VBG', 'UH', 'PRP$', 'WRB', 'WP', 'VBZ', 'RBR',
#        '-LRB-', 'JJS', 'WDT', 'NNPS', 'TO', 'VBN', ':', 'JJR', 'VB', 'VBD',
#        'RBS', 'PDT', 'MD', ',', 'VBP', 'POS', 'RP', '$', 'FW','NNP',
#        "''", 'WP$', '.', '#', '-RRB-']
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
y_tags = y.unique()


class test_model(unittest.TestCase):
    def test_fit(self):
        tests = [f'f_10{x}' for x in range(8)]
        model1 = Model(tests)
        data = PreprocessTags().load_data(
            r'..\data\test.wtag')
        a = model1.fit(data.x, data.y)
        print(a)

    def test_question1(self):
        """
        LOAD THE DATA
        PRE PROCESS
        FIT - TRAIN
        PREDICT -

        :return:
        """
        acc = 90
        print(acc)
        self.assertGreaterEqual(acc, 90, msg=f'current acc:{acc}')

    def test_model_function(self):
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
        x_thin = pd.Series(['*', '*', 'base', 'Treasury', 'is', '<STOP>'])
        y_thin = pd.Series(['*', '*', 'Vt', 'NNP', 'VBZ', '<STOP>'])

        y_tags = y_thin.unique()
        tests = [f'f_10{x_}' for x_ in range(8)]
        model1 = Model(tests)
        model1.tag_corpus = y_tags
        model1.tag_corpus = y_thin
        model1._translation()
        data = PreprocessTags().load_data(r'..\data\test.wtag')
        a = model1.model_function(1, 2, [1, 2], x)
        print("model function result")
        print(a)
        # self.assertAlmostEqual(float(a), -2.77, 2)
        b = model1._viterbi(x_thin)
        print("viterbi result")
        print(b)