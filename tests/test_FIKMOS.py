import unittest
from models.prerocesing import PreprocessTags
import numpy as np
import pandas as pd

from models.sentence_processor import FinkMos

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


class test_rapnaparkhi(unittest.TestCase):

    def test_features(self):
        r = FinkMos(x, y, tests=[f'f_10{x}' for x in range(8)], tag_corpus=y_tags)
        print()
        print(r.linear_loss())

    def test_fs2(self):
        r = FinkMos(x, y, tests=[f'f_10{x}' for x in range(8)], tag_corpus=y_tags)
        print()
        print(r.to_feature_space2(3, 'DT', 'NNP', 'VBZ'))

    def test_normalize(self):
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
        r = FinkMos(x, y, tests=[f'f_10{x}' for x in range(8)], tag_corpus=y_tags)
        value = r.sentence_non_lineard_loss(np.zeros([8]))
        self.assertAlmostEqual(value, 72.0873, 3)

    def test_tuples2tensor(self):
        fm = FinkMos(x, y, y_tags)

        filename = r'alex_shor.h5'
        tuple_mat = np.load(filename)
        # print(tuple_mat.shape)
        # tuple_mat =
        result = fm.create_feature_tensor(tuple_mat, batch_size=10000)

        print(result.shape())

    def test_create_tuples(self):
        data = PreprocessTags(True).load_data(
            r'..\data\train.wtag')
        tag_corp = pd.Series(data.y[0:50000]).unique()
        fm = FinkMos(data.x[0:50000], data.y[0:50000], tag_corp)
        # fm = FinkMos(x, y, y_tags)
        fm.create_tuples()
        print("fm.weight_mat")
        print(fm.weight_mat)
        print("fm.tuple_5_list")
        print(fm.tuple_5_list)
        fm.create_feature_sparse_list_v2()
        print(len(fm.f_matrix_list))
        print(fm.f_matrix_list[0].shape)
