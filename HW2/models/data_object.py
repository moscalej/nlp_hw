# imports
from collections import namedtuple


#


# DP_sentence = namedtuple("DP_sentence", "graph, sentence, tags, f")
class DP_sentence:
    def __init__(self, sentence, tags, graph=None, ):  # TODO validate about tags
        self.graph_tag = graph  # {1: [2], 2: [1, 3],3: [1]}
        self.graph_est = None  # {1: [2], 2: [1, 3],3: [1]}
        self.sentence = sentence
        self.tags = tags
        self.f = None
        self.train = True if graph is not None else False  # TODO see if useful
