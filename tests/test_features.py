import unittest
from models.sentence_processor import FinkMos
import pandas as pd
from models.features import Features
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
f_dict = Features().get_tests()
class test_rapnaparkhi(unittest.TestCase):

    def test_rapna_100_107(self):
        r = FinkMos(x, y, tests=[f'f_10{x}' for x in range(8)], tag_corpus=y_tags)
        print('')
        print(r.x[6], r.y[6])
        self.failUnlessEqual(f_dict['f_100'](r.x,6, r.y.loc[6], r.y.loc[5], r.y.loc[4]), 0)
        # Test 101
        print(r.x[24], r.y[24])
        self.failUnlessEqual(f_dict['f_101'](r.x,24, r.y.loc[24], r.y.loc[23], r.y.loc[22]), 1)
        print(r.x[4], r.y[4])
        self.failUnlessEqual(f_dict['f_101'](r.x,4, r.y.loc[4], r.y.loc[3], r.y.loc[2]), 0)

        # Test 102
        print(r.x[25], r.y[25])
        self.failUnlessEqual(f_dict['f_102'](r.x,25, r.y.loc[25], r.y.loc[24], r.y.loc[23]), 1)
        print(r.x[4], r.y[4])
        self.failUnlessEqual(f_dict['f_102'](r.x,4, r.y.loc[4], r.y.loc[3], r.y.loc[2]), 0)

        # Test 103
        place = 7
        place2 = 3
        print(r.y[[place, place - 1, place - 2]])
        self.failUnlessEqual(f_dict['f_103'](r.x,place, r.y.loc[place], r.y.loc[place - 1], r.y.loc[place - 2]), 0)
        print(r.x[4], r.y[4])
        self.failUnlessEqual(f_dict['f_103'](r.x,place2, r.y.loc[place2], r.y.loc[place2 - 1], r.y.loc[place2 - 2]), 0)

        # Test 104
        place = 5
        place2 = 6
        print(r.y[[place, place - 1]])
        self.failUnlessEqual(f_dict['f_104'](r.x,place, r.y.loc[place], r.y.loc[place - 1], r.y.loc[place - 2]), 0)
        print(r.x[4], r.y[4])
        self.failUnlessEqual(f_dict['f_104'](r.x,place2, r.y.loc[place2], r.y.loc[place2 - 1], r.y.loc[place2 - 2]), 0)

        # Test 105
        place = 26
        place2 = 6
        print(r.y[place], 105)
        self.failUnlessEqual(f_dict['f_105'](r.x,place, r.y.loc[place], r.y.loc[place - 1], r.y.loc[place - 2]), 1)
        print(r.y[place2])
        self.failUnlessEqual(f_dict['f_105'](r.x,place2, r.y.loc[place2], r.y.loc[place2 - 1], r.y.loc[place2 - 2]), 0)

        # Test 106
        place = 26
        place2 = 6
        print(r.y[place])
        self.failUnlessEqual(f_dict['f_106'](r.x,place, r.y.loc[place], r.y.loc[place - 1], r.y.loc[place - 2]), 0)
        print(r.y[place2])
        self.failUnlessEqual(f_dict['f_106'](r.x,place2, r.y.loc[place2], r.y.loc[place2 - 1], r.y.loc[place2 - 2]), 0)

        # Test 107
        place = 26
        place2 = 6
        print(r.y[place])
        self.failUnlessEqual(f_dict['f_107'](r.x,place, r.y.loc[place], r.y.loc[place - 1], r.y.loc[place - 2]), 0)
        print(r.y[place2])
        self.failUnlessEqual(f_dict['f_107'](r.x,place2, r.y.loc[place2], r.y.loc[place2 - 1], r.y.loc[place2 - 2]), 0)
