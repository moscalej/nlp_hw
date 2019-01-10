import unittest

import numpy as np
import pandas as pd

import models.features as feat
from models.features import Features
from models.prerocesing import PreprocessTags
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

    def test_create_tuples(self):
        data = PreprocessTags(True).load_data(
            r'..\data\train.wtag')
        word_num = 1_000
        tag_corp = pd.Series(data.y[0:word_num]).unique()
        # generate tests - (comment out if file is updated)
        feat_generator = Features()
        feat_generator.generate_tuple_corpus(data.x[0:word_num], data.y[0:word_num])
        for template in feat.templates_dict.values():
            feat_generator.generate_lambdas(template['func'], template['tuples'])
        feat_generator.save_tests()

        fm = FinkMos(data.x[0:word_num], data.y[0:word_num], tag_corp)
        fm.create_tuples()
        print("fm.weight_mat")
        print(fm.weight_mat)
        print("fm.tuple_5_list")
        print(fm.tuple_5_list)
        fm.create_feature_sparse_list_v2()
        # print(len(fm.f_matrix_list))
        print(fm.f_matrix_list[0].shape)
        fm.minimize_loss()
        fm.v.dump('values')
