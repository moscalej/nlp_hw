# imports
import numpy as np
# import models.boot_camp as bc
from models.data_object import DP_sentence
from models.chu_liu import Digraph
from models.boot_camp import BootCamp


#


class DP_Model:

    def __init__(self, boot_camp, w=None):
        # assert isinstance(boot_camp, BootCamp)
        self.w = w
        self.bc = boot_camp  # defines feature space
        self.lr = 1  # TODO

    def fit(self, obj_list, epochs, truncate=0):
        """

        :param obj_list:
        :param epochs:
        :param truncate:
        :return: should this function return any statistics
        """
        self.bc.investigate_soldiers(obj_list)
        if truncate > 0:  # TODO: review boot camp usage flow
            self.bc.truncate_features(truncate)
        else:
            self.bc.features.tokenize()
        print(f"Training model with {self.bc.features.num_features} features")
        self.w = np.zeros(self.bc.features.num_features)
        # self.w = np.random.rand(self.bc.features.num_features)
        self.bc.train_soldiers(obj_list)  # create f_x for each
        generator_f_x = (obj.f for obj in obj_list)  # TODO: Generator
        # generator_f_x = [obj.f for obj in obj_list]  # TODO: Generator
        # TODO: make sure passing an argument like this is really by pointer
        generator_y = (obj.graph_tag for obj in obj_list)
        # generator_y = [obj.graph_tag for obj in obj_list]
        #
        self.perceptron(generator_f_x, generator_y, epochs)

    def predict(self, obj_list):
        """

        :param x: iterable list of sentences
        :type x:
        :return:
        :rtype:
        """
        self.bc.train_soldiers(obj_list)  # create f_x for each
        for obj in obj_list:
            full_graph, weight_dict = self.create_full_graph(obj.f)
            get_score = lambda i, j: weight_dict[i, j]
            obj.graph_est = Digraph(full_graph, get_score=get_score).mst().successors
        result = [obj.graph_est for obj in obj_list]
        return result

    def score(self, obj_list):
        self.predict(obj_list)
        total = len(obj_list)
        corret = 0
        for obj in obj_list:
            isinstance(obj, DP_sentence)
            corret += 1 if obj.graph_est == obj.graph_tag else 0
        return corret / total

    def perceptron(self, f_x_list, y, epochs):
        for epo in range(epochs):
            for (f_x, graph) in zip(f_x_list, y):
                full_graph, weight_dict = self.create_full_graph(f_x)

                # def get_score(k, l):
                #     return weight_dict[k, l]

                # get_score = lambda k, l: weight_dict[k, l]
                opt_graph = Digraph(full_graph, get_score=lambda k, l: weight_dict[k, l]).mst().successors
                opt_graph = {key: value for key, value in opt_graph.items() if value}  # remove empty
                if opt_graph != graph:
                    diff = self.graph2vec(graph, f_x) - self.graph2vec(opt_graph, f_x)
                    self.w = self.w + diff
                else:
                    print(f"over - fit on {epo} index")
                    pass

    def create_full_graph(self, f_x):
        """
        Create full graph and weighted matrix for chu liu
        :param f_x: feature space of edges in sentence
        :type f_x: list of sparse matrices
        :return: full_graph and weighted matrix
        :rtype:
        """
        # f_x dims: list of #{edge_source} slices of #{edge_target} x #{features} (edge source dim = edge_target dim but only in src 0 is valid [root])
        full_graph = {src: range(1, len(f_x)) for src in range(len(f_x))}  # TODO: save in dictionary
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
