import time
import unittest
from scipy import sparse as spar
from models.dp_model import DP_Model
import numpy as np


class test_model(unittest.TestCase):
    def test_create_full_graph(self):
        # make fake tensor for input:
        # shared baseline
        fake_tens = []
        dim_src = 4
        dim_trg = dim_src - 1
        dim_feat = 7
        for trgt_feat_slice in range(dim_src):
            t = spar.random(dim_trg, dim_feat, density=0.25)  # sparse dot
            fake_tens.append(t)
        dummy_model = DP_Model(dim_feat, None, w=np.ones(dim_feat))
        #

        full_graph, weight_mat = dummy_model.create_full_graph(fake_tens)
        print(full_graph)
        print(weight_mat)

    def test_graph2vec(self):
        # shared baseline
        fake_tens = []
        dim_src = 4
        dim_trg = dim_src
        dim_feat = 7
        for trgt_feat_slice in range(dim_src):
            t = spar.random(dim_trg, dim_feat, density=0.25).tocsr()  # sparse dot
            fake_tens.append(t)
        dummy_model = DP_Model(dim_feat, None, w=np.ones(dim_feat))
        dummy_graph = {0: [2], 1: [3], 2: [1]}
        #
        graph_w = dummy_model.graph2vec(dummy_graph, fake_tens)
        print(graph_w)
