import time
import unittest
from scipy import sparse as spar

from models.Preprocess import PreProcess
from models.dp_model import DP_Model
from models.data_object import DP_sentence
from models.boot_camp import BootCamp, Features
import numpy as np
from scipy.sparse import csr_matrix

t_f_1 = csr_matrix([[1,0,0,1],[1,0,0,1],[1,0,0,1]])
t_f = [t_f_1,t_f_1,t_f_1]
ds = DP_sentence(['hola','tu','mama'],['tt','tt','tt'])
ds.f = t_f
ds.graph = {0: [0, 1, 2], 1: [], 2: []}


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

    def test_predict(self):
        par = PreProcess(r'../data/toy.labeled')
        bc = BootCamp(Features())
        ds_list = par.parser()
        model = DP_Model(num_features=4, boot_camp=bc)  # TODO do we need number of fatures
        result = model.predict(ds_list)
        print(result)
        # self.assertAlmostEqual(result,[{0: [0, 1, 2], 1: [], 2: []}, {0: [0, 1, 2], 1: [], 2: []}])

    def test_fit(self):
        par = PreProcess(r'../data/toy.labeled')
        bc = BootCamp(Features())
        ds_list = par.parser()
        model = DP_Model(num_features=4, boot_camp=bc)  # TODO do we need number of fatures
        result = model.fit(ds_list, 3)
        print(result)


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
