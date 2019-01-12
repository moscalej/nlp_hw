# imports
import numpy as np
# import models.boot_camp as bc
from models.data_object import DP_sentence
from models.chu_liu import Digraph
#


class DP_Model:

    def __init__(self, num_features, boot_camp, tagger, w=None):
        self.w = np.zeros(num_features) if w is None else w
        self.bc = boot_camp  # defines feature space
        self.tagger = tagger  # taging function
        assert (num_features, w.shape[0])  # make sure w has right dims
        self.lr = 1  # TODO
        pass

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

                if opt_graph != graph:  # TODO: currently only pseudo
                    self.w = self.w + self.lr * (graph - opt_graph)  # TODO: see how to define this substraction

    def create_full_graph(self, f_x):
        """
        Create full weighted graph for chu liu
        :param f_x:
        :type f_x: tensor
        :return: full_graph and weighted matrix
        :rtype:
        """
        # f_x dims: list of #{edge_source} slices of #{edge_target} x #{features} (edge source = edge_target +1 [root])
        full_graph = {src: range(f_x[0].shape[0]) for src in range(len(f_x))}
        results = []
        for src_feat_slice in f_x:
            t = src_feat_slice.dot(self.w)  # sparse dot
            results.append(t)
        weight_mat = np.array(results)
        return full_graph, weight_mat
        pass