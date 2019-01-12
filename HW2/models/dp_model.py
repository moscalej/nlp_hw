# imports
import numpy as np
# import models.boot_camp as bc
from models.data_object import DP_sentence
from models.chu_liu import Digraph


#


class DP_Model:

    def __init__(self, num_features, boot_camp, w=None):
        self.w = np.zeros(num_features) if w is None else w
        self.bc = boot_camp  # defines feature space
        self.lr = 1  # TODO

    def fit(self, obj_list, epochs):
        self.bc.train_soldiers(obj_list)  # create f_x for each
        f_x = (obj.f for obj in obj_list)  # TODO: Generator
        # TODO: make sure passing an argument like this is really by pointer
        y = (obj.graph for obj in obj_list)
        #
        self.perceptron(f_x, y, epochs)

    def predict(self, obj_list):
        """

        :param x: iterable list of sentences
        :type x:
        :return:
        :rtype:
        """
        # obj_list = []
        # for ind, sentence in enumerate(x):
        #     obj_list.append(DP_sentence(sentence=sentence, tags=tags[ind]))
        self.bc.train_soldiers(obj_list)  # create f_x for each
        for obj in obj_list:
            full_graph, weight_dict = self.create_full_graph(obj.f)
            get_score = lambda i, j: weight_dict[i, j]
            obj.graph = Digraph(full_graph, get_score=get_score).greedy().successors
        result = [obj.graph for obj in obj_list]
        return result

    def perceptron(self, f_x_list, y, epochs):
        for i in range(epochs):
            for (f_x, graph) in zip(f_x_list, y):
                full_graph, weight_dict = self.create_full_graph(f_x)
                get_score = lambda i, j: weight_dict[i, j]
                opt_graph = Digraph(full_graph,
                                    get_score=get_score).greedy().successors

                if opt_graph != graph:
                    self.w = self.w + self.lr * (self.graph2vec(graph, f_x) - self.graph2vec(opt_graph, f_x))

    def create_full_graph(self, f_x):
        """
        Create full graph and weighted matrix for chu liu
        :param f_x: feature space of edges in sentence
        :type f_x: list of sparse matrices
        :return: full_graph and weighted matrix
        :rtype:
        """
        # f_x dims: list of #{edge_source} slices of #{edge_target} x #{features} (edge source dim = edge_target dim but only in src 0 is valid [root])
        full_graph = {src: range(f_x[0].shape[0]) for src in range(len(f_x))}
        results = []
        for trgt_feat_slice in f_x:
            t = trgt_feat_slice.dot(self.w)  # sparse dot
            results.append(t)
        weight_mat = np.array(results)
        return full_graph, weight_mat

    def graph2vec(self, graph, f_x):
        """
        returns a vector describing the contribution of each feature to the given graph weight
        :param graph:
        :type graph:
        :param f_x:
        :type f_x:
        :return: vector each entry is the sum of all the edges given feature contribution
        :rtype: np.array vector of w dims
        """
        test_weigh_vec = np.zeros(self.w.shape[0])
        for key, vals in graph.items():
            for val in vals:
                # key is the source index of the edge and val is the target index
                test_weigh_vec += f_x[key][val, :]
        return test_weigh_vec
