import numpy as np
import pandas as pd
from scipy import sparse as spar
from scipy.optimize import minimize

from models.features import Features


class FinkMos:

    def __init__(self, x, y, tag_corpus):
        assert isinstance(x, pd.Series)
        self.tag_corpus = tag_corpus
        self.test_dict = Features().get_tests()
        self.test_vec = np.array([test['func'][1] for test in self.test_dict.values()])
        self.x = x
        self.y = y
        self.f_matrix_list = None  #
        self.linear_loss_done = None
        # self.word2number = {word: index for index, word in enumerate(x.value_counts().index)}
        # tc = tag_corpus.shape[0]
        self.fast_test = dict()
        self.fast_predict = dict()
        self.weight_mat = None
        self.tuple_5_list = None
        self.tup5_2index = dict()
        self.opt = None
        self.v = None
        self.f_v_train = None
        self.calc_from_mem = None

    def create_tuples(self):
        """
        tuple handling
        :Create: tuple_5_list (list of 5 tuple combinations)
            weight_mat (number of occurences of each tuple in dataset)
        :return:
        :rtype:
        """
        tx_0 = self.x.values
        ty_0 = self.y.values
        tx_1 = np.roll(tx_0, 1)
        tx_2 = np.roll(tx_1, 1)
        ty_1 = np.roll(ty_0, 1)
        ty_2 = np.roll(ty_1, 1)

        tuple_6_np = ty_0 + "_" + ty_1 + "_" + ty_2 + "_" + tx_0 + "_" + tx_1 + "_" + tx_2
        tuple_6_counts_series = pd.Series(tuple_6_np).value_counts()
        tuple_5_df = pd.DataFrame([ty_1, ty_2, tx_0, tx_1, tx_2]).T
        tuple_5_df.sort_values([0, 1, 2, 3, 4], inplace=True)  # sort the 5 tuple_list by
        tuple_5_df.drop_duplicates(inplace=True, keep='first')  # remove duplicates
        self.tuple_5_list = list(map(lambda x: list(x[1]), tuple_5_df.iterrows()))  # make list of every row of the DF

        #  create wight mask
        weight_mask = spar.csr_matrix((self.tag_corpus.shape[0], tuple_5_df.shape[0]), dtype=int)
        self.tup5_2index = {"_".join(x): num for num, x in enumerate(tuple_5_df.values)}
        for tup, count in tuple_6_counts_series.items():
            tup_0 = tup.split('_')[0]
            tup_5 = '_'.join(tup.split('_')[1:])
            ind_j = self.tup5_2index[tup_5]
            itemindex = np.where(self.tag_corpus == tup_0)
            ind_i = itemindex[0]
            weight_mask[ind_i, ind_j] = count
        self.weight_mat = weight_mask

    def create_feature_sparse_list_v2(self, training_fm=None):
        # return a list of sparse matrices, each matrix
        # tuple_5_list = self.tuple_5_list

        tuple_5_size = len(self.tuple_5_list)
        # tuple_0_list = self.tag_corpus  # [[elem1], [elem2], ...] ->
        tuple_0_size = self.tag_corpus.shape[0]
        num_test = len(self.test_dict)
        # returns a list of empty spars matrices
        result = [spar.csr_matrix((tuple_5_size, num_test), dtype=bool) for _ in range(tuple_0_size)]
        # iterate list of test names
        if self.y is None:  # inference mode
            calculated = spar.csr_matrix((tuple_5_size, tuple_0_size), dtype=int)
            for tup_5_ind, tup5 in enumerate(self.tuple_5_list):
                # if calculated before take value
                tup_5_str = ('_').join(tup5)
                if tup_5_str in training_fm.tup5_2index:
                    ind_in_train = training_fm.tup5_2index[tup_5_str]
                    calculated[tup_5_ind, :] = training_fm.f_v_train[:, ind_in_train]
                    continue
                for tup_0_ind, tup0 in enumerate(self.tag_corpus):
                    tup = (tup0,) + tuple(tup5)
                    result[tup_0_ind][tup_5_ind, :] = np.array([test(tup) for test in self.test_vec])
            self.calc_from_mem = calculated
        else:
            for test_ind, (key, val) in enumerate(self.test_dict.items()):
                # iterate list of tuples per test
                for tup in set(val['tup_list']):
                    tup_0_ind = np.where(tup[0] == self.tag_corpus)[0][0]
                    tup_5_ind = self.tup5_2index['_'.join(tup[1:])]
                    result[tup_0_ind][tup_5_ind, test_ind] = True
        self.f_matrix_list = result

    def loss_function(self, v):
        f_v = self.dot(v)  # add factor
        f_v_mask = self.weight_mat.multiply(f_v)
        l_fv = np.sum(np.sum(f_v_mask))  # * mask
        exp_ = np.exp(f_v)
        exp_sum = np.sum(exp_, axis=0)
        repetitions = np.array(self.weight_mat.sum(axis=0))  # from here not sparse
        ln = np.log(exp_sum) * repetitions
        sum_ln = np.sum(ln)
        return sum_ln - l_fv #+ 0.1 * np.linalg.norm(v)

    def loss_gradient(self, v):
        f_v = self.dot(v)  # dims: tup_0 x tup5
        e_f_v = np.exp(f_v)  # dims: tup0 x tup5
        z = np.sum(e_f_v, axis=1) + 1e-11  # dims: tup0 x tup5
        p = (e_f_v.T / z).T  # dims: tup0 x tup5
        f_p_tup5_list = []  # sum over tuples list
        f_v_tup_0_tests = []
        for tup_0_ind, sparse_matrix in enumerate(self.f_matrix_list):
            spar_t = sparse_matrix.T
            # Left
            weight_vec = self.weight_mat[tup_0_ind, :]
            weighted_slice = spar.csr_matrix.multiply(spar_t, weight_vec)
            f_v_tests = weighted_slice.sum(axis=1)
            f_v_tup_0_tests.append(f_v_tests)

            # Right
            f_p = spar.csr_matrix.multiply(spar_t, p[tup_0_ind, :])  # dims: tup5 x tests
            f_p_tup5_list.append(f_p)

        sparce_list = sum(f_p_tup5_list)
        sparce_list_w_weight = spar.csr_matrix.multiply( sparce_list , self.weight_mat.sum(axis=0))
        right =np.squeeze( np.array(sparce_list_w_weight.sum(axis=1)))

        left = np.array(f_v_tup_0_tests)  # dims 1 X dim(V)
        left_sum = np.squeeze(np.array(np.sum(left, axis=0)))
        regularization = 0.2 * v
        result = left_sum - right# - regularization
        neg_result = - result
        return neg_result


    def dot(self, v):
        results = []
        for sparce_matrix in self.f_matrix_list:
            t = sparce_matrix.dot(v)
            results.append(t)
        return np.array(results)

    def minimize_loss(self):
        self.opt = minimize(self.loss_function,
                            np.ones(len(self.test_dict)),
                            # jac=self.loss_gradient,
                            options=dict(disp=True,
                                         maxiter=15,
                                         # eps=1e-5,
                                         # gtol= 1e-6
                                         ),
                            method='CG',
                            callback=self.callback_cunf)
        self.v = self.opt.x
        self.f_v_train = self.dot(self.v)

    def callback_cunf(self, x):
        print(f'Current loss {self.loss_function(x)}')



    def prob_q2(self, v, y_token, training_fm):
        self.create_feature_sparse_list_v2(training_fm)  # creates f_matrix_list
        f_v = self.dot(v) + self.calc_from_mem.T  # dims tup0 x tup5
        y_nomin = np.array(f_v[y_token])  # dims tup5 x 1
        exp_ = np.array(np.exp(f_v)).squeeze()
        exp_sum = np.sum(exp_, axis=0)  # dims tup5 x 1
        prob = np.array(y_nomin / (exp_sum+1e-10))[0]  # dims tup5 x 1
        return prob
