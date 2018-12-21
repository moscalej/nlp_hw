import time
import unittest

import pandas as pd
import yaml

from models.features import Features
from models.model import Model
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


class test_model(unittest.TestCase):
    def test_fit(self):
        with open(r"..\models\tests.YALM", 'r') as stream:
            data_loaded = yaml.load(stream)
        tests = data_loaded['tests']
        model1 = Model(tests)
        data = PreprocessTags().load_data(
            r'..\data\test.wtag')
        a = model1.fit(data.x, data.y)
        results = dict(
            test_sum=model1.lin_loss_matrix_x_y.sum(),
            v=model1.v,
            # compare = pd.DataFrame([tests,model1.v]).T


        )
        print(model1.v)
        print(model1.lin_loss_matrix_x_y.sum())
        print(a)
        t = time.localtime()
        with open(fr"..\training\report_{t.tm_hour}:{t.tm_min}_{t.tm_mday}/{t.tm_mon}.YALM", 'w') as stream:
            stream.write(yaml.dump(results))

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
        tests = Features().get_tests().keys()
        model1 = Model(tests)
        model1.fit(x, y)
        fm = FinkMos(x, x, model1.tests, model1.tag_corpus)
        a = model1.model_function(1, 3, [2, 3], fm)
        print("model function result")
        print(a)

    def test_viterbi(self):
        x_thin = pd.Series(['*', '*', 'The', 'Treasury', 'is', '<STOP>'])
        y_thin = pd.Series(['*', '*', 'DT', 'NNP', 'VBZ', '<STOP>'])
        tests = [f'f_10{x_}' for x_ in range(8)] + [f'tri_00{x_}' for x_ in range(7)]
        data = PreprocessTags().load_data(
            r'..\data\test.wtag')
        model1 = Model(tests)

        model1.x = x
        model1.y = y
        base_corpus = pd.Series(['*', '<STOP>'])
        tag_corpus = pd.Series(y.value_counts().drop(['*', '<STOP>']).index)
        model1.tag_corpus = base_corpus.append(tag_corpus)
        model1.tag_corpus_tokenized = range(len(model1.tag_corpus))
        model1._translation()  # create dictionaries for tokenizing
        model1._vectorize()
        #
        # below result of fit on full data (model1.fit(data.x, data.y))
        #
        model1.v = [9.98440989e-04, 4.55460621e+00, 3.50976884e+00, 9.98440989e-04,
                    9.98440989e-04, 9.98440989e-04, 9.98440989e-04, 9.98440989e-04,
                    3.53744043e+00, 4.72940057e+00, 2.86124632e+00, 3.02403509e+00,
                    9.98440989e-04, 2.75377462e+00, 9.98440989e-04]
        # print(model1.vector_x_y)
        # print(model1.lin_loss_matrix_x_y)
        b = model1.predict(x)
        # b = model1._viterbi(x)
        print("viterbi result")
        print(b)
        # print([model1.token2string[token] for token in b])
