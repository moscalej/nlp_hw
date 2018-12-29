import pickle
import unittest

import dill as pickle
import pandas as pd

import features as feat
from models.features import Features
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
y_tags = y.unique()


class features(unittest.TestCase):

    def test_feature_generator(self):
        data = PreprocessTags(True).load_data(
            r'..\data\toy_dataset.txt')
        feat_generator = Features()
        feat_generator.generate_tuple_corpus(data.x[0:10000], data.y[0:10000])
        try:
            # feat_generator.get_tests()  # loads last version saved
            pass
        except:
            pass
        for template in feat.templates_dict.values():
            feat_generator.generate_lambdas(template['func'], template['tuples'])
        # feat_generator.add_lambdas(feat.suffix_funcs_all)  # DONE
        # feat_generator.add_lambdas(feat.prefix_funcs_all)  # DONE
        result = feat_generator.lambdas
        print(len(result))
        with open(fr"../training/report_lambdas_dict.p", 'wb') as stream:
            pickle.dump(result, stream)
