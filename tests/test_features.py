from models.prerocesing import PreprocessTags
import pickle
import dill as pickle
import unittest
from models.sentence_processor import FinkMos
import pandas as pd
from models.features import Features
import features as feat
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
            r'..\data\train.wtag')
        feat_generator = Features(data.x, data.y)
        try:
            with open(fr"../training/report_lambdas_dict.p", 'rb') as stream:
                feat_generator.add_lambdas(pickle.load(stream))
        except:
            pass
        for template in feat.templates_dict.values():
            print(template)
            feat_generator.generate_lambdas(template['func'], template['tuples'])
        # feat_generator.add_lambdas(feat.suffix_funcs_all)  # DONE
        # feat_generator.add_lambdas(feat.prefix_funcs_all)  # DONE
        result = feat_generator.lambdas
        print(len(result))
        with open(fr"../training/report_lambdas_dict.p", 'wb') as stream:
            pickle.dump(result, stream)
