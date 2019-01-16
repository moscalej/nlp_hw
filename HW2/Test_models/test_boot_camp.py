import time
import unittest
from scipy import sparse as spar
from models.dp_model import DP_Model
from models.data_object import DP_sentence
from models.boot_camp import BootCamp
import numpy as np
from scipy.sparse import csr_matrix
from models.boot_camp import Features

# import sympy; sympy.init_printing()

t_f_1 = csr_matrix([[1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]])
t_f = [t_f_1, t_f_1, t_f_1]
ds = DP_sentence(['hola', 'tu', 'mama'], ['tt', 'tt', 'tt'])
ds.f = t_f
ds.graph_tag = {0: [0, 1, 2], 1: [], 2: []}


class test_Preprocess(unittest.TestCase):
    def test_init(self):
        feat = Features()

    def test_extract_features(self):
        feat = Features()
        feat.extract_features(ds)
        print(feat.features.keys())

    def test_truncate_features(self):
        feat = Features()
        feat.extract_features(ds)
        feat.truncate_features(5)
        print("num of keys in updated:")
        print(len(list(feat.features.keys())))

    def test_fill_tensor(self):
        feat = Features()
        feat.extract_features(ds)
        feat.truncate_features(5)
        feat.fill_tensor(ds)
        print([mat.toarray() for mat in ds.f])  # for printing
