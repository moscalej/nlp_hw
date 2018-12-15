import unittest
from models.temp import *
class test_model(unittest.TestCase):
    def test_test1(self):
        weights = 10
        feature_num = 10
        num_tags = 10
        weights = np.random.rand(feature_num)
        model = DummyModel(weights, q_func, feature_factory)
        result = viterbi(model, sentence=["the", "dog", "barks"], all_tags=range(num_tags))

    def test_viterbi(self):