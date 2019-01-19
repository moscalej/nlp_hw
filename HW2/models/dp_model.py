# imports
import time

from HW2.models.data_object import DP_sentence
from HW2.models.chu_liu import Digraph
from models.boot_camp import BootCamp
from numba import njit
import numpy as np
from pandas import DataFrame
import pandas as pd


#


class DP_Model:
    """
    Dp Model is a semantic parcing model, witch trains using the perceptron alghorith and
    chu_liu maximon spaning tree
    """

    def __init__(self, boot_camp, w=None):
        print(type(boot_camp))
        assert isinstance(boot_camp, BootCamp)
        self.w = w
        self.bc = boot_camp  # defines feature space
        self.lr = 1  # TODO

    def fit(self, obj_list, epochs, truncate=0, validation=None):
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
        self.w = np.zeros(self.bc.features.num_features)
        # Give each Sentence object a Tensor witch will be use to predict
        self.bc.train_soldiers(obj_list)  # create f_x for each

        return self.perceptron(obj_list, epochs, validation=validation)

    def predict(self, obj_list):
        """
        Given a list array of Dp sentence wiill fill the predicted graph to each object

        Parameters
        -----
        :param x: iterable list of DP_sentence
        :type x: list, array
        :return: a list of dictionaries contanion graph relentions
        :rtype: list (dict)
        """
        self.bc.train_soldiers(obj_list)  # create f_x for each
        for obj in obj_list:
            full_graph, weight_dict = self.create_full_graph(obj.f, obj)
            graph_est = Digraph(full_graph, get_score=lambda k, l: weight_dict[k, l]).mst().successors
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
                initial_graph, weight_dict = self.create_full_graph(f_x, obj_list[ind])
                graph_est = Digraph(initial_graph, get_score=lambda k, l: weight_dict[k, l]).mst().successors
                graph_est = {key: value for key, value in graph_est.items() if value}  # remove empty
                if not compare_graph_fast(list(graph_est.items()), list(graph_tag.items())):
                    diff = self.graph2vec(graph_tag, f_x) - self.graph2vec(graph_est, f_x)
                    self.w = self.w + diff
                else:
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
            print(f'Finish base model with {epo} epochs at {time.strftime("%X %x")} t_acc{train_acc}')
        return pd.DataFrame(results_all, columns=['Model', 'time', 'epochs', 'train_score', 'val_score', 'n_features'])

    def score(self, obj_list):
        self.predict(obj_list)
        total = len(obj_list)
        corret = 0
        for obj in obj_list:
            isinstance(obj, DP_sentence)
            corret += 1 if compare_graph_fast(list(obj.graph_est.items()), list(obj.graph_tag.items())) else 0
        return corret / total

    def create_full_graph(self, f_x, obj):
        """
        Create full graph and weighted matrix for chu liu

        Parameters
        -------
        :param f_x: feature space of edges in sentence
        :type f_x: list of sparse matrices
        :return: full_graph and weighted matrix
        :rtype: dict, np.array
        """
        feature_obj = self.bc.features
        # full_graph = {src: range(1, len(f_x)) for src in range(len(f_x))}  # TODO: save in dictionary
        full_graph = {}
        debug_count = 0
        for src in range(len(f_x)):
            full_graph[src] = []
            for trg in range(len(f_x)):
                if feature_obj.features[feature_obj.get_key(f'tag_src_tag_trg', obj.tags[src], obj.tags[trg])]:
                    full_graph[src].append(trg)  # TODO: save in dictionary
                    debug_count += 1
        print(f"Created {debug_count} edges instead of {len(f_x)*len(f_x)}")
        results = []
        if self.w.min() == 0 and self.w.max() == 0:
            return full_graph, np.zeros((len(f_x), len(f_x)))
        for trgt_feat_slice in f_x:
            t_d = np.array(trgt_feat_slice.toarray())
            t_d_w = t_d @ self.w
            t = trgt_feat_slice.dot(self.w)  # sparse dot
            results.append(t)
        weight_mat = np.array(results)
        return full_graph, weight_mat

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
        for key, vals in graph.items():
            for val in vals:
                # key is the source index of the edge and val is the target index
                test_weigh_vec += f_x[key].A[val, :]  # TODO: consider sum of sparse matrix
        return test_weigh_vec

    def get_model(self):
        return self.bc.get_model()


# @njit()
def compare_graph_fast(graph_est, graph_tag):
    """

    :param graph_est:
    :type graph_est: list
    :param graph_tag:
    :type graph_tag: list
    :return:
    """
    graph_est.sort()
    graph_tag.sort()
    for i in range(len(graph_est)):
        if graph_est[i][0] != graph_tag[i][0]:
            return False
        if set(graph_est[i][1]) != set(graph_tag[i][1]):
            return False
    return True
