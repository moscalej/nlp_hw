# imports
import time

from HW2.models.data_object import DP_sentence
from HW2.models.chu_liu import Digraph
from HW2.models.boot_camp import BootCamp
from numba import njit
import numpy as np
from pandas import DataFrame
import pandas as pd
from heapq import nlargest
import scipy.sparse as spar


#


class DP_Model:
    """
    Dp Model is a semantic parcing model, witch trains using the perceptron alghorith and
    chu_liu maximon spaning tree
    """

    def __init__(self, boot_camp, w=None):
        print(type(boot_camp))
        # assert isinstance(boot_camp, BootCamp)
        self.w = w  # TODO make sure sparse
        self.bc = boot_camp  # defines feature space

    def fit(self, obj_list, epochs, truncate=0, validation=None, fast=False):
        """
        Trains the model using the [perceptron alghorith](https://en.wikipedia.org/wiki/Perceptron)
        and chu liu.

        Parameters
        ----------
        :param obj_list: list DP_MOdel
        :type obj_list: DP_Model list

        :param epochs: number of epochs to train the Model
        :type epochs: int
        :param truncate: max number of features to use
        :type truncate: int
        :return: should this function return any statistics

        Logic flow
        ----------
            - Create tests
            - Pick the truncate most important tests
            - Give each Sentence object a Tensor witch will be use to predict
            - Use the perceptron Algorith to train the model

        """
        # Create tests
        self.bc.investigate_soldiers(obj_list)
        # Pick the truncate most important tests
        if truncate > 0:  # TODO: review boot camp usage flow
            self.bc.truncate_features(truncate)
        else:
            self.bc.features.tokenize()
        print(f"Training model with {self.bc.features.num_features} features")
        #  TODO: if w was loaded don't init w
        self.w = np.zeros(self.bc.features.num_features)
        # self.w = spar.coo_matrix((self.bc.features.num_features, 1), dtype=np.float64)

        # Give each Sentence object a Tensor witch will be use to predict
        self.bc.train_soldiers(obj_list, fast=fast)  # create f_x for each

        return self.perceptron(obj_list, epochs, validation=validation)

    def predict(self, obj_list, fast=False):
        """
        Given a list array of Dp sentence wiill fill the predicted graph to each object

        Parameters
        -----
        :param x: iterable list of DP_sentence
        :type x: list, array
        :return: a list of dictionaries contanion graph relentions
        :rtype: list (dict)
        """
        self.bc.train_soldiers(obj_list, fast=fast)  # create f_x for each
        for obj in obj_list:
            # full_graph, weight_dict = self.create_full_graph(obj.f, obj)
            # initial_graph = self.create_init_graph(obj)
            edge_weights = self.create_edge_weights(obj.f)
            # initial_graph = self.keep_top_edges(obj, edge_weights, n_top=10)
            initial_graph = obj.graph_est
            graph_est = Digraph(initial_graph, get_score=lambda k, l: edge_weights[k, l]).mst().successors
            obj.graph_est = {key: value for key, value in graph_est.items() if value}  # remove empty
        result = [obj.graph_est for obj in obj_list]
        return result

    def perceptron(self, obj_list, epochs, validation=None):
        """
        Same perceptron algorith from the Tirgul
        :param f_x_list: List of tensors
        :type f_x_list: list
        :param y: list
        :param epochs:
        :return: Data frame with the results of the training
        :rtype : DataFrame
        """
        f_x_list = [obj.f for obj in obj_list]
        y = [obj.graph_tag for obj in obj_list]
        start_time = time.time()
        results_all = []
        test_acc = 0
        total = len(f_x_list)
        for epo in range(epochs):
            current = 0
            for ind, (f_x, graph_tag) in enumerate(zip(f_x_list, y)):
                edge_weights = self.create_edge_weights(f_x)
                if epo in [0, 1, 2]:
                    n_top = max(int(len(f_x) / 8), 3)
                    initial_graph = self.keep_top_edges(obj_list[ind], edge_weights, n_top=n_top)
                else:
                    initial_graph = obj_list[ind].graph_est
                graph_est = Digraph(initial_graph, get_score=lambda k, l: edge_weights[k, l]).mst().successors
                graph_est = {key: value for key, value in graph_est.items() if value}
                # TODO Debug Remove
                # diff_pre = self.graph2vec(graph_tag, f_x) - self.graph2vec(graph_est, f_x)
                # if not compare_graph_fast(graph_est, graph_tag):  # TODO: I think there is some bug here, I get better overfit without it
                # if np.sum(diff_pre - diff):
                #     pass
                # if not compare_graph_fast(graph_est, graph_tag) != np.sum(diff):
                #     pass
                # if is_zero_diff != is_same_graphs and is_same_graphs is True:
                #     print("Bug")
                # TODO Debug Remove

                diff = self.graph2vec(graph_tag, f_x) - self.graph2vec(graph_est, f_x)
                is_zero_diff = bool(np.sum(diff) == 0)
                if not is_zero_diff:
                    self.w = self.w + diff
                else:
                    is_same_graphs = compare_graph_fast(graph_est, graph_tag)
                    if is_same_graphs:
                        current += 1

            if validation is not None:
                test_acc = self.score(validation)
            train_acc = current / total
            results_all.append([self.get_model(),
                                time.time() - start_time,
                                epo,
                                train_acc,
                                test_acc,
                                self.bc.features.num_features])
            print(f'Finished {epo} epoch for base model at {time.strftime("%X %x")} train_acc{train_acc}')
        return pd.DataFrame(results_all, columns=['Model', 'time', 'epochs', 'train_score', 'val_score', 'n_features'])

    def score(self, obj_list):
        self.predict(obj_list)
        total = len(obj_list)
        correct = 0
        for obj in obj_list:
            isinstance(obj, DP_sentence)
            correct += 1 if compare_graph_fast(obj.graph_est, obj.graph_tag) else 0
        return correct / total

    # def create_init_graph(self, obj):
    #     feature_obj = self.bc.features
    #     # full_graph = {src: range(1, len(f_x)) for src in range(len(f_x))}  # TODO: save in dictionary
    #     full_graph = {}
    #     debug_count = 0
    #     for src in range(len(obj.f)):
    #         full_graph[src] = []
    #         for trg in range(len(obj.f)):
    #             if feature_obj.features[feature_obj.get_key(f'tag_src_tag_trg', obj.tags[src], obj.tags[trg])]:
    #                 full_graph[src].append(trg)  # TODO: save in dictionary
    #                 debug_count += 1
    #     print(f"Created {debug_count} edges instead of {len(obj.f)*len(obj.f)}")
    #     return full_graph

    def create_edge_weights(self, f_x):
        """
        Create full graph and weighted matrix for chu liu

        Parameters
        -------
        :param f_x: feature space of edges in sentence
        :type f_x: list of sparse matrices
        :return: full_graph and weighted matrix
        :rtype: dict, np.array
        """

        results = []
        if self.w.min() == 0 and self.w.max() == 0:
            return np.zeros((len(f_x), len(f_x)))
        for trgt_feat_slice in f_x:
            t = trgt_feat_slice.dot(self.w)  # sparse dot
            results.append(t)
        weight_mat = np.array(results)
        return weight_mat

    def keep_top_edges(self, obj, edge_weights, n_top=10):
        new_graph = {}
        for src, trgs in obj.graph_est.items():
            new_graph[src] = nlargest(n_top, trgs, key=lambda j: edge_weights[src, j])
        return new_graph

    def graph2vec(self, graph, f_x):
        """
        returns a vector describing the contribution of each feature to the given graph weight
        :param graph:
        :type graph: dict()
        :param f_x:
        :type f_x:
        :return: vector each entry is the sum of all the edges given feature contribution
        :rtype: np.array vector of w dims
        """
        test_weigh_vec = np.zeros(self.w.shape[0])
        # test_weigh_vec = spar.csr_matrix((1, self.w.shape[0]), dtype=np.int64)
        for src, trgs in graph.items():
            slice_indices = f_x[src].indices
            slice_ptr = f_x[src].indptr
            for trg in trgs:
                activ_feat_inds = slice_indices[slice_ptr[trg]:slice_ptr[trg + 1]]
                # activ_feat_inds = list(f_x[src].getrow(trg).indices)
                for feat_ind in activ_feat_inds:
                    test_weigh_vec[feat_ind] += 1
                # np.add.at(test_weigh_vec, activ_feat_inds, 1)

            # activ_feat_inds = [f_x[src].col[ind] for ind, val in enumerate(f_x[src].row) if val in trgs]
            # # np.add.at(test_weigh_vec, activ_feat_inds, 1)
            # for feat_ind in activ_feat_inds:
            #     test_weigh_vec[feat_ind] += 1
        return test_weigh_vec

    def get_model(self):
        return self.bc.get_model()


# @njit()
def compare_graph_fast(graph_est, graph_tag):
    """

    :param graph_est_:
    :type graph_est_: list
    :param graph_tag_:
    :type graph_tag_: list
    :return:
    """
    graph_est_ = list(graph_est.items())
    graph_tag_ = list(graph_tag.items())
    if len(graph_est_) != len(graph_tag_):
        return False
    graph_est_.sort()
    graph_tag_.sort()
    for i in range(len(graph_est_)):
        if graph_est_[i][0] != graph_tag_[i][0]:
            return False
        if set(graph_est_[i][1]) != set(graph_tag_[i][1]):
            return False

    return True
