# imports
import time

from HW2.models.data_object import DP_sentence
from HW2.models.chu_liu import Digraph
from models.boot_camp import BootCamp
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

    def __init__(self, boot_camp: BootCamp, w=None):
        # assert isinstance(boot_camp, BootCamp)
        self.w = w  # TODO make sure sparse
        self.bc = boot_camp  # defines feature space

    def fit(self, obj_list, epochs, truncate_top=0, truncate_bottom=None, validation=None, fast=True):
        """
        Trains the model using the [perceptron alghorith](https://en.wikipedia.org/wiki/Perceptron)
        and chu liu.

        Parameters
        ----------
        :param validation: Subset of the dataset witch is use for validating the  Tests
        :type validation: list
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
        if truncate_top > 0:  # TODO: Cahnge to remoce

            # self.bc.features.truncate_by_thresh(truncate)
            self.bc.truncate_features(n_top=truncate_top, n_bottom=truncate_bottom)
        else:
            self.bc.features.tokenize()
        print(f"Training model with {self.bc.features.num_features} features")
        #  TODO: if w was loaded don't init w
        self.w = np.zeros(self.bc.features.num_features)
        # self.w = spar.coo_matrix((self.bc.features.num_features, 1), dtype=np.float64)

        # Give each Sentence object a Tensor witch will be use to predict
        self.bc.train_soldiers(obj_list, fast=fast)  # create f_x for each

        return self.perceptron(obj_list, epochs, validation=validation)

    def predict(self, obj_list, fast=False, epoch=0):
        """
        Given a list array of Dp sentence wiill fill the predicted graph to each object

        Parameters
        -----
        :param x: iterable list of DP_sentence
        :type x: list, array
        :return: a list of dictionaries contanion graph relentions
        :rtype: list (dict)
        """
        if epoch == 0:
            self.bc.train_soldiers(obj_list, fast=fast)  # create f_x for each

        for obj in obj_list:
            edge_weights = self.create_edge_weights(obj.f)
            initial_graph = obj.full_graph.copy()
            graph_est = Digraph(initial_graph, get_score=lambda k, l: edge_weights[k, l]).mst().successors
            obj.graph_est = {key: value for key, value in graph_est.items() if value}  # remove empty
        result = [obj.graph_est for obj in obj_list]
        return result

    def perceptron(self, obj_list: list, epochs: int, validation: list = None) -> pd.DataFrame:
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
                # if epo not in [0, 1, 2]:
                if False:
                    n_top = max(int(len(f_x) - epo), 5)
                    initial_graph = self.keep_top_edges(obj_list[ind], edge_weights, n_top=n_top)
                else:
                    initial_graph = obj_list[ind].full_graph
                graph_est = Digraph(initial_graph.copy(), get_score=lambda k, l: edge_weights[k, l]).mst().successors
                graph_est = {key: value for key, value in graph_est.items() if value}
                diff = self.graph2vec(graph_tag, f_x) - self.graph2vec(graph_est, f_x)
                is_zero_diff = bool(np.sum(diff) == 0)
                is_same_graphs = compare_graph_fast(graph_est, graph_tag)
                # if not is_zero_diff and epo in list(range(10)) or not is_same_graphs:
                if not is_zero_diff:
                    lr = 0.0 if is_same_graphs else 1
                    # if not is_same_graphs:
                    self.w = self.w + lr * diff
                else:
                    if is_same_graphs:
                        current += 1

            if validation is not None:
                # print('Doing VAl')
                test_acc = self.score(validation, epoch=epo)
            train_acc = current / total
            results_all.append([self.get_model(),
                                time.time() - start_time,
                                epo,
                                train_acc,
                                test_acc,
                                self.bc.features.num_features])
            print(
                f'Finished {epo} epoch for base model at {time.time() - start_time} train_acc {train_acc} Test {test_acc}')
        return pd.DataFrame(results_all, columns=['Model', 'time', 'epochs', 'train_score', 'val_score', 'n_features'])

    def score(self, obj_list, epoch=0):
        self.predict(obj_list, epoch=epoch)
        return accuracy(obj_list)

    def create_edge_weights(self, f_x):
        """
        Create full graph and weighted matrix for chu liu

        Parameters
        -------
        :param f_x: feature space of edges in sentence
        :type f_x: list of sparse matrices
        :return: full_graph and weighted matrix
        :rtype:  np.array
        """

        results = []
        if self.w.min() == 0 and self.w.max() == 0:
            return np.zeros((len(f_x), len(f_x)))
        for trgt_feat_slice in f_x:
            t = trgt_feat_slice.dot(self.w)  # sparse dot
            results.append(t)
        weight_mat = np.array(results)
        return weight_mat

    def keep_top_edges(self, obj: DP_sentence, edge_weights: np.ndarray, n_top: int = 10) -> dict:
        """

        :param obj:
        :param edge_weights:
        :param n_top:
        :return:
        """
        new_graph = {}
        for src, trgs in obj.graph_est.items():
            new_graph[src] = nlargest(n_top, trgs, key=lambda j: edge_weights[src, j])
        return new_graph

    def graph2vec(self, graph: dict, f_x: list) -> np.ndarray:

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
        for src, trgs in graph.items():
            slice_indices = f_x[src].indices
            slice_ptr = f_x[src].indptr
            for trg in trgs:
                activ_feat_inds = slice_indices[slice_ptr[trg]:slice_ptr[trg + 1]]
                for feat_ind in activ_feat_inds:
                    test_weigh_vec[feat_ind] += 1
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


def accuracy(iter_1: list) -> np.ndarray:
    """

    :param iter_1: List of sentence objects
    :type iter_1: list
    :return: accuaracy
    :rtype: float
    """
    res =[]
    for so in iter_1:
        res.append(so2df(so))

    return np.mean(np.array(res),axis=-1)


def so2df(so: DP_sentence) -> np.ndarray:  # Sentence Object
    """
    Converts Sentence objects to a Data frame with the same format
    as the hw

    Params
    ----

    :param so: Sentence Object from witch we take the information
    :type so: DP_sentence

    :return: A data frame containing the information
    :rtype: np.ndarray
    """
    th = np.zeros(so.sentence.shape[0], dtype=np.int8)
    for key, value in so.graph_est.items():
        th[value] = key

    th_2 = np.zeros(so.sentence.shape[0], dtype=np.int8)
    for key_2, value_2 in so.graph_tag.items():
        th_2[value_2] = key_2
    return np.mean(th ==th_2)
