import numpy as np
import pandas as pd
from scipy import sparse as spar
from scipy.optimize import minimize

from models.features import Features


class FinkMos:

    def __init__(self, x, y, tag_corpus):
        assert isinstance(x, pd.Series)
        assert isinstance(y, pd.Series)
        self.tag_corpus = tag_corpus
        self.test_dict = Features().get_tests()
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

    def create_feature_sparse_list_v2(self):
        # return a list of sparse matrices, each matrix
        # tuple_5_list = self.tuple_5_list
        tuple_5_size = len(self.tuple_5_list)
        # tuple_0_list = self.tag_corpus  # [[elem1], [elem2], ...] ->
        tuple_0_size = self.tag_corpus.shape[0]
        num_test = len(self.test_dict)
        # returns a list of empty spars matrices
        result = [spar.csr_matrix((tuple_5_size, num_test), dtype=bool) for _ in range(tuple_0_size)]
        # iterate list of test names
        for test_ind, (key, val) in enumerate(self.test_dict.items()):
            # iterate list of tuples per test
            for tup in set(val['tup_list']):
                tup_0_ind = np.where(tup[0] == self.tag_corpus)[0][0]
                tup_5_ind = self.tup5_2index['_'.join(tup[1:])]
                result[tup_0_ind][tup_5_ind, test_ind] = True
        self.f_matrix_list = result

    def loss_function(self, v, batch_size=4096):
        sum_ln_tot = 0
        l_fv_tot = 0
        for batch_low in range(0, self.weight_mat.shape[0], batch_size):
            batch_high = batch_low + batch_size
            f_v = self.dot(v, batch_low, batch_high)  # add factor
            f_v_mask = self.weight_mat[:, batch_low:batch_high].multiply(f_v)
            l_fv = np.sum(np.sum(f_v_mask))  # * mask
            exp_ = np.exp(f_v)
            exp_sum = np.sum(exp_, axis=0)
            repetitions = np.array(self.weight_mat[:, batch_low:batch_high].sum(axis=0))  # from here not sparse
            ln = np.log(exp_sum) * repetitions
            sum_ln = np.sum(ln)
            sum_ln_tot += sum_ln
            l_fv_tot += l_fv
        return sum_ln_tot - l_fv_tot + 0.1 * np.linalg.norm(v)

    def loss_gradient(self, v, batch_size=4096):
        left_sum_tot = np.zeros([len(self.test_dict)])
        right_tot = np.zeros([len(self.test_dict)])
        for batch_low in range(0, self.weight_mat.shape[0], batch_size):
            batch_high = batch_low + batch_size

            f_v = self.dot(v, batch_low, batch_high)  # add factor
            # dims: tup_0 x tup5
            # (self.weight_mat.sum(axis=0) * tup_0_tests).sum(axis=0)
            e_f_v = np.exp(f_v)  # dims: tup0 x tup5
            z = np.sum(e_f_v, axis=1)  # dims: tup0 x tup5
            p = (e_f_v.T / z).T  # dims: tup0 x tup5
            f_p_tup5_list = []  # sum over tuples list
            f_v_tup_0_tests = []
            for tup_0_ind, sparse_matrix in enumerate(self.f_matrix_list):
                spar_t = sparse_matrix[batch_low:batch_high].T
                f_p = spar.csr_matrix.multiply(spar_t, p[tup_0_ind, :])  # dims: tup5 x tests
                weight_vec = self.weight_mat[tup_0_ind, batch_low:batch_high]
                weighted_slice = spar.csr_matrix.multiply(spar_t, weight_vec)
                f_v_tests = weighted_slice.sum(axis=1)
                f_v_tup_0_tests.append(f_v_tests)
                f_p_tup5 = spar.csr_matrix.multiply(f_p, weight_vec)
                f_p_tup5_sum = f_p_tup5.sum(axis=1)
                f_p_tup5_list.append(f_p_tup5_sum)
            left = np.squeeze(np.array(f_v_tup_0_tests))  # dims 1 X dim(V)
            left_sum = np.sum(left, axis=0)
            f_p_tup5_arr = np.squeeze(np.array(f_p_tup5_list))
            right = f_p_tup5_arr.sum(axis=0)
            left_sum_tot += left_sum
            right_tot += right
        regularization = 0.2 * v
        result = left_sum_tot - right_tot - regularization
        neg_result = -result  # for minimization
        return neg_result

    def dot(self, v, batch_low, batch_high):
        results = []
        for sparce_matrix in self.f_matrix_list:
            t = sparce_matrix[batch_low:batch_high, :].dot(v)
            results.append(t)
        return np.array(results)

    def minimize_loss(self):
        self.opt = minimize(self.loss_function,
                            np.ones(len(self.test_dict)),
                            jac=self.loss_gradient,
                            options=dict(disp=True, maxiter=10),
                            method='CG',
                            callback=self.callback_cunf)
        self.v = self.opt.x

    def callback_cunf(self, x):
        print(f'Current loss {self.loss_function(x)}')

    def softmax_denominator(self, v, history_i, y, y_1, y_2):

        hash_name = f"{self.x[history_i]}{y_1}{y_2}"
        if hash_name in self.fast_predict:
            e_val = self.fast_predict[hash_name]
        else:
            results = []
            for tag in self.tag_corpus:
                temp = self.to_feature_space2(history_i, tag, y_1, y_2)
                if len(temp) > 0:
                    results.append(temp)
            f_matrix = pd.Series(results)
            v_f = f_matrix.apply(lambda x: np.sum(v[x]))
            e_val = np.sum(np.exp(v_f)) + self.tag_corpus.size - v_f.shape[0]
            self.fast_predict[hash_name] = e_val
        return e_val

    def prob_q(self, v, history_i, y, y_1, y_2):
        features = self.to_feature_space2(history_i, y, y_1, y_2)
        features_v = np.exp(np.sum(v[features]))
        return features_v / self.softmax_denominator(v, history_i, y, y_1, y_2)
