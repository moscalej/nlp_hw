import time
import unittest
from scipy import sparse as spar

from models.Preprocess import PreProcess
from models.dp_model import DP_Model
from models.data_object import DP_sentence
from models.boot_camp import BootCamp, Features
import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
t_f_1 = csr_matrix([[1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]])
t_f = [t_f_1, t_f_1, t_f_1]
ds = DP_sentence(['hola', 'tu', 'mama'], ['tt', 'tt', 'tt'])
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
        model = DP_Model(boot_camp=bc)
        result = model.predict(ds_list)
        print(result[0])
        # self.assertAlmostEqual(result,[{0: [0, 1, 2], 1: [], 2: []}, {0: [0, 1, 2], 1: [], 2: []}])

    def test_fit(self):
        par = PreProcess(r'../data/toy.labeled')
        bc = BootCamp(Features())
        ds_list = par.parser()
        model = DP_Model(boot_camp=bc)
        model.fit(ds_list, epochs=50)
        results = model.score(ds_list)
        # print(model.w)
        clean_est = {key: value for key, value in results[0].items() if value}  # remove empty
        print(f"Predicted: {clean_est}")
        sorted_ground_truth = dict(sorted(ds_list[0].graph_tag.items()))
        print(f"Ground Truth: {sorted_ground_truth}")


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

    def test_main(self):
        NUM_EPOCHS = [10]
        MODELS = ['base', 'advance']
        NUMBER_OF_FEATURES = [500, 5000, 50000, 100_000, 0]
        toy_path = r'C:\Users\afinkels\Desktop\private\Technion\Master studies\עיבוד שפה טבעית\HW\hw_repo\nlp_hw\HW2\data\toy.labeled'
        toy_10__train_path = r'C:\Users\afinkels\Desktop\private\Technion\Master studies\עיבוד שפה טבעית\HW\hw_repo\nlp_hw\HW2\data\toy_10_train.labeled'
        toy_5__train_path = r'C:\Users\afinkels\Desktop\private\Technion\Master studies\עיבוד שפה טבעית\HW\hw_repo\nlp_hw\HW2\data\toy_5_train.labeled'
        toy_10_test_path = r'C:\Users\afinkels\Desktop\private\Technion\Master studies\עיבוד שפה טבעית\HW\hw_repo\nlp_hw\HW2\data\toy_10_test.labeled'
        train_path = r'C:\Users\afinkels\Desktop\private\Technion\Master studies\עיבוד שפה טבעית\HW\hw_repo\nlp_hw\HW2\data\train.labeled'
        test_path = r'C:\Users\afinkels\Desktop\private\Technion\Master studies\עיבוד שפה טבעית\HW\hw_repo\nlp_hw\HW2\data\test.labeled'
        DATA_PATH = toy_10_test_path
        TEST_PATH = toy_path
        RESULTS_PATH = r'C:\Users\afinkels\Desktop\private\Technion\Master studies\עיבוד שפה טבעית\HW\hw_repo\nlp_hw\HW2\Test_models'
        results_all = []

        data = PreProcess(DATA_PATH).parser()
        test = PreProcess(TEST_PATH).parser()
        # BASE MODEL
        bc = BootCamp(Features('bas'))
        model = DP_Model(boot_camp=bc)
        for n_epochs in NUM_EPOCHS:
            start_time = time.time()
            model.fit(data, epochs=n_epochs, fast=False, truncate=0)
            train_acc = model.score(data)
            test_acc = model.score(test)
            results_all.append(
                ['base', time.time() - start_time, n_epochs, train_acc, test_acc, bc.features.num_features])
            print(
                f'Finish base model with {n_epochs} epochs at {time.strftime("%X %x")} train_acc{train_acc} and test_acc{test_acc}')
        df_results = pd.DataFrame(results_all,
                                  columns=['Model', 'time', 'epochs', 'train_score', 'val_score', 'n_features'])
        df_results.to_csv(f'{RESULTS_PATH}\\from_test_re_{time.localtime()}.csv')
