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
class test_rapnaparkhi(unittest.TestCase):

    def test_rapna_100_107(self):
        r = FinkMos(x, y, tests=[f'f_10{x}' for x in range(8)], tag_corpus=y_tags)
        print('')
        print(r.x[6], r.y[6])
        self.failUnlessEqual(r.f_100(6, r.y.loc[6], r.y.loc[5], r.y.loc[4]), 0)
        # Test 101
        print(r.x[24], r.y[24])
        self.failUnlessEqual(r.f_101(24, r.y.loc[24], r.y.loc[23], r.y.loc[22]), 1)
        print(r.x[4], r.y[4])
        self.failUnlessEqual(r.f_101(4, r.y.loc[4], r.y.loc[3], r.y.loc[2]), 0)

        # Test 102
        print(r.x[25], r.y[25])
        self.failUnlessEqual(r.f_102(25, r.y.loc[25], r.y.loc[24], r.y.loc[23]), 1)
        print(r.x[4], r.y[4])
        self.failUnlessEqual(r.f_102(4, r.y.loc[4], r.y.loc[3], r.y.loc[2]), 0)

        # Test 103
        place = 7
        place2 = 3
        print(r.y[[place, place - 1, place - 2]])
        self.failUnlessEqual(r.f_103(place, r.y.loc[place], r.y.loc[place - 1], r.y.loc[place - 2]), 0)
        print(r.x[4], r.y[4])
        self.failUnlessEqual(r.f_103(place2, r.y.loc[place2], r.y.loc[place2 - 1], r.y.loc[place2 - 2]), 0)

        # Test 104
        place = 5
        place2 = 6
        print(r.y[[place, place - 1]])
        self.failUnlessEqual(r.f_104(place, r.y.loc[place], r.y.loc[place - 1], r.y.loc[place - 2]), 0)
        print(r.x[4], r.y[4])
        self.failUnlessEqual(r.f_104(place2, r.y.loc[place2], r.y.loc[place2 - 1], r.y.loc[place2 - 2]), 0)

        # Test 105
        place = 26
        place2 = 6
        print(r.y[place], 105)
        self.failUnlessEqual(r.f_105(place, r.y.loc[place], r.y.loc[place - 1], r.y.loc[place - 2]), 1)
        print(r.y[place2])
        self.failUnlessEqual(r.f_105(place2, r.y.loc[place2], r.y.loc[place2 - 1], r.y.loc[place2 - 2]), 0)

        # Test 106
        place = 26
        place2 = 6
        print(r.y[place])
        self.failUnlessEqual(r.f_106(place, r.y.loc[place], r.y.loc[place - 1], r.y.loc[place - 2]), 0)
        print(r.y[place2])
        self.failUnlessEqual(r.f_106(place2, r.y.loc[place2], r.y.loc[place2 - 1], r.y.loc[place2 - 2]), 0)

        # Test 107
        place = 26
        place2 = 6
        print(r.y[place])
        self.failUnlessEqual(r.f_107(place, r.y.loc[place], r.y.loc[place - 1], r.y.loc[place - 2]), 0)
        print(r.y[place2])
        self.failUnlessEqual(r.f_107(place2, r.y.loc[place2], r.y.loc[place2 - 1], r.y.loc[place2 - 2]), 0)

    def test_run_line(self):
        r = FinkMos(x, y, tests=[f'f_10{x}' for x in range(8)], tag_corpus=y_tags)
        a = r.run_line_tests('f_101')
        # self.assertEqual(1,1)

    def test_features(self):
        r = FinkMos(x, y, tests=[f'f_10{x}' for x in range(8)], tag_corpus=y_tags)
        print()
        print(r.fill_test())

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
        self.assertAlmostEqual(value, 69.3147, 3)

