import time
import unittest
from scipy import sparse as spar
from models.dp_model import DP_Model
import numpy as np

class test_model(unittest.TestCase):
    def test_create_full_graph(self):
        # make fake tensor for input:
        fake_tens = []
        dim_src = 4
        dim_trg = dim_src - 1
        dim_feat = 7
        for trgt_feat_slice in range(dim_src):
            t = spar.random(dim_trg, dim_feat, density=0.25)  # sparse dot
            fake_tens.append(t)
        #
        dummy_model = DP_Model(dim_feat, None, w=np.ones(dim_feat))
        full_graph, weight_mat = dummy_model.create_full_graph(fake_tens)
        print(full_graph)
        print(weight_mat)
