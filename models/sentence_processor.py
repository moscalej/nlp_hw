import numpy as np
import pandas as pd
from scipy import sparse as spar

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
        weight_mask = np.zeros([self.tag_corpus.shape[0], tuple_5_df.shape[0]])
        self.tup5_2index = {"_".join(x): num for num, x in enumerate(tuple_5_df.values)}
        for tup, count in tuple_6_counts_series.items():
            tup_0 = tup.split('_')[0]
            tup_5 = '_'.join(tup.split('_')[1:])
            ind_j = self.tup5_2index[tup_5]
            itemindex = np.where(self.tag_corpus == tup_0)
            ind_i = itemindex[0]
            weight_mask[ind_i, ind_j] = count
        self.weight_mat = weight_mask

    def create_feature_tensor(self, tuple_matrix, batch_size):
        dims = (tuple_matrix.shape[0], tuple_matrix.shape[2], batch_size)

        # result = np.empty(dims, np.bool_, )
        result = []
        for ind, key in enumerate(list(self.test_dict.items())[0:batch_size]):
            if "pre" in key[0] or "suf" in key[0]:
                continue

            # result[:, :, ind] = np.apply_along_axis(test, 1, tuple_matrix)
            result.append(np.apply_along_axis(key[1], 1, tuple_matrix))
            # result[:, :, ind] = np.array(list(map(test, tuple_matrix)))
            # result[:, :, ind] = np.vectorize(test)(tuple_matrix)
            if ind % 50 == 0:
                print(ind)
        return np.array(result)

    def create_feature_sparse_list_v1(self, tuple_matrix):
        # return a list of sparse matrices, the ith matrix is the results of the ith test
        result = []
        for ind, key in enumerate(list(self.test_dict.items())):
            if "pre" in key[0] or "suf" in key[0]:
                continue
            result.append(spar.csr_matrix(np.apply_along_axis(key[1], 1, tuple_matrix)))
            if ind % 50 == 0:
                print(ind)
        return result

    def create_feature_sparse_list_v2(self):
        # return a list of sparse matrices, each matrix
        # tuple_5_list = self.tuple_5_list
        tuple_5_size = len(self.tuple_5_list)
        # tuple_0_list = self.tag_corpus  # [[elem1], [elem2], ...] ->
        tuple_0_size = self.tag_corpus.shape[0]
        num_test = len(self.test_dict)
        # returns a list of empty spars matrices
        result = [spar.csr_matrix((num_test, tuple_5_size), dtype=bool) for _ in range(tuple_0_size)]
        # iterate list of test names
        for test_ind, (key, val) in enumerate(self.test_dict.items()):
            # iterate list of tuples per test
            for tup in set(val['tup_list']):
                tup_0_ind = np.where(tup[0] == self.tag_corpus)[0][0]
                tup_5_ind = self.tup5_2index['_'.join(tup[1:])]
                result[tup_0_ind][test_ind, tup_5_ind] = True
        # decode tuple to indices

        # for ind, tup_0 in enumerate(tuple_0_list):  # per test one vector per tuple
        #     for key, test in self.test_dict.items():  # mat per test
        #         if "pre" in key[0] or "suf" in key[0]:  # TODO: relevant?
        #             continue
        #         spar_mat = spar.csr_matrix((num_test, tuple_5_size), dtype=bool)
        #         tup_list = tuple_5_list.copy()
        #         list(map(lambda x: x.insert(0, tup_0), tup_list))  # create tuple list for tup0
        #         # spar_mat[ind, :] = spar.csr_matrix((list(map(test, tup_list))))
        #
        #         result.append(spar_mat)
        #         if ind % 50 == 0:
        #             print(ind)
        self.f_matrix_list = result

    def linear_loss(self, v):
        """

        :param tests:
        :return: Return vector where each value is the value of a test
        """
        feaures = []
        if self.linear_loss_done is None:
            for i in range(3, self.x.size):
                temp = self.to_feature_space2(i, self.y[i], self.y[i - 1], self.y[i - 2])
                if len(temp) > 0:
                    feaures.append(temp)
            self.linear_loss_done = pd.Series(feaures)
        loss = self.linear_loss_done.apply(lambda x: np.sum(v[x]))
        return np.sum(loss)

    def sentence_non_linear_loss_inner(self, v, h_word_i):
        y_1, y_2 = self.y[h_word_i - 1], self.y[h_word_i - 2]
        if self.f_matrix_list[h_word_i] is None:
            results = []
            for tag in self.tag_corpus:
                temp = self.to_feature_space2(h_word_i, tag, y_1, y_2)
                if len(temp) > 0:
                    results.append(temp)
            f_matrix = pd.Series(results)
            self.f_matrix_list[h_word_i] = f_matrix
        v_f = self.f_matrix_list[h_word_i].apply(lambda x: np.sum(v[x]))
        e_val = np.sum(np.exp(v_f)) + self.tag_corpus.size - v_f.shape[0]
        return e_val

    def sentence_non_lineard_loss(self, v):
        values = []
        for word in range(2, self.x.shape[0]):
            values.append(self.sentence_non_linear_loss_inner(v, word))
        logvals = np.log(np.array(values))
        sentence_normal = np.sum(logvals)
        return sentence_normal

    def to_feature_space2(self, history_i, y, y_1, y_2):
        hash_name = f"{self.x[history_i]}{y}{y_1}{y_2}"
        if hash_name in self.fast_test:
            results = self.fast_test[hash_name]
        else:
            results = []
            for index, test in enumerate(self.tests):
                if self.test_f[test](self.x, history_i, y, y_1, y_2) == 1:
                    results.append(index)
            self.fast_test[hash_name] = results
        return results

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


class CustomFeatures:
    def __init__(self, x, y):
        assert isinstance(x, pd.Series)
        assert isinstance(y, pd.Series)
        self.x = x
        self.y = y
