import time
import unittest
from scipy import sparse as spar
from models.dp_model import DP_Model
from models.data_object import DP_sentence
from models.boot_camp import BootCamp
from HW2.models.Preprocess import PreProcess
import numpy as np
from scipy.sparse import csr_matrix
from models.boot_camp import Features

# import sympy; sympy.init_printing()

t_f_1 = csr_matrix([[1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]])
t_f = [t_f_1, t_f_1, t_f_1]
ds = DP_sentence(['hola', 'tu', 'mama'], ['tt', 'tt', 'tt'])
ds.f = t_f
ds.graph_tag = {0: [0, 1, 2], 1: [], 2: []}


class test_features(unittest.TestCase):
    def test_init(self):
        feat = Features()
        bc =BootCamp(feat)

    def test_extract_features(self):
        feat = Features()
        par = PreProcess(r'../data/toy.labeled')
        bc = BootCamp(feat)
        bc.investigate_soldiers(par.parser())


    def test_truncate_features(self):
        feat = Features()
        par = PreProcess(r'../data/toy.labeled')
        bc = BootCamp(feat)
        bc.investigate_soldiers(par.parser())
        bc.truncate_features(10)

    def test_fill_tensor(self):
        feat = Features()
        par = PreProcess(r'../data/toy.labeled')
        bc = BootCamp(feat)
        soldiers = par.parser()
        bc.investigate_soldiers(soldiers)
        # bc.truncate_features(10)
        bc.train_soldiers(soldiers)

