import unittest
from models.temp import *
from models.model import *


class test_model(unittest.TestCase):
    def test_test1(self):
        weights = 10
        feature_num = 10
        num_tags = 10
        weights = np.random.rand(feature_num)
        model = DummyModel(weights, q_func, feature_factory)
        result = viterbi(model, sentence=["the", "dog", "barks"], all_tags=range(num_tags))

    def test_test1(self):
        tests = [f'f_10{x}' for x in range(8)]
        model1 = Model(tests)
        model1.tag_corpus_tokenized = range(5)
        answer = model1.predict(["*", "*", "the", "dog", "barks", "<STOP>"])
        print(answer)