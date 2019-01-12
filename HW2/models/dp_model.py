# imports
import numpy as np
# import models.boot_camp as bc
from models.data_object import DP_sentence
from models.chu_liu import Digraph
#


class DP_Model:

    def __init__(self, num_features, boot_camp, tagger, w=None):
        w = np.random.rand(num_features) if w is None else w
        self.bc = boot_camp  # defines feature space
        self.tagger = tagger  # taging function
        assert (num_features, w.shape[0])  # make sure w has right dims
        self.lr
        pass

    def fit(self, objs, epochs):
        f_x = [obj.f for obj in objs]  # TODO: make sure passing an argument like this is really by pointer
        y = [obj.graph for obj in objs]
        #
        self.perceptron(f_x, y, epochs)

    def predict(self, x):
        """

        :param x: iterable list
        :type x:
        :return:
        :rtype:
        """
        tags = self.tagger(x)
        obj_list = []
        for ind, sentence in enumerate(x):
            obj_list.append(DP_sentence(sentence=sentence, tags=tags[ind]))
        self.bc.train_soldiers(x)  # create f_x for each
        for obj in obj_list:
            weighted_graph = self.create_full_tree(obj.f)
            obj.graph = Digraph(
                weighted_graph).greedy().successors  # TODO: currently only pseudo apply chu liu to object
        result = [obj.graph for obj in obj_list]
        return result

    def perceptron(self, f_x_list, y, epochs):
        for i in range(epochs):
            for ind, f_x in enumerate(f_x_list):
                weighted_graph = self.create_full_graph(f_x)
                opt_graph = Digraph(weighted_graph).greedy().successors  # TODO: currently only pseudo

                if opt_graph != y[ind]:  # TODO: currently only pseudo
                    self.w = self.w + self.lr * (y[ind] - opt_graph)  # TODO: see

    def create_full_graph(self, f_x):
        """
        Create full weighted graph for chu liu
        :param f_x:
        :type f_x: tensor
        :return: full weighted graph
        :rtype:
        """
        # self.w
        pass