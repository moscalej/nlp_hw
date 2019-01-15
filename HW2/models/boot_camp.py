# imports
from collections import defaultdict

#

# TODO: feature definition:
# TODO remember we don't have output value restriction (can be a real number)
"""
Features flow:
Features is a DefaultDict, holding a count of the times this feature was seen

Creation of features, given a data_obj parse all edges in the obj, 
and enter the context and tagging to templates to add new keys to dict.

Truncate_features iterate over all given features in dict and return the n most seen features

Create key2token dict 

Graph_to_feature_tensor: given an object iterate over full graph edges,
for each edge create keys by templates and fill their values in the tensor
"""


class Features:
    def __init__(self):
        features = defaultdict(int)
        features['bias'] = 1
        self.key2token = dict()

    def extract_features(self, data_obj):
        """
        Creates keys in dict, a key corresponds to a feature
        The key holds the count that the feature was seen
        :param data_obj:
        :type data_obj:
        :return:
        :rtype:
        """
        graph = data_obj.graph_tag
        context = data_obj.sentence
        tags = data_obj.tags
        for src_ind, trg_inds in graph.items():
            for trg_ind in trg_inds:  # edge in the graph (src_ind, trg_ind)
                # current
                keys = self.get_keys(self, src_ind, trg_ind, context, tags)
                self._add_keys(keys)

    def Truncate_features(self, n):
        """
        Truncate n most freuquent features
        :param n:
        :type n:
        :return:
        :rtype:
        """
        self.features = dict(sorted(self.features, key=self.features.get, reverse=True)[:n])
        self.key2token = dict(enumerate(self.features.keys()))

    def fill_tensor(self, data_obj):
        context = data_obj.sentence
        tags = data_obj.tags
        num_nodes = len(tags)
        # graph = get_full_graph(num_nodes)
        graph = {src: range(1, num_nodes) for src in range(num_nodes)}
        for src_ind, trg_inds in graph.items():
            for trg_ind in trg_inds:  # edge in the graph (src_ind, trg_ind)
                keys = self.get_keys(self, src_ind, trg_ind, context, tags)
                exist = self._check_keys(keys)
                activ_feat_inds = self.key2token[exist]
                data_obj[src_ind][trg_ind, activ_feat_inds] = 1

    def get_keys(self, src_ind, trg_ind, context, tags):
        src_word = context[src_ind]
        trg_word = context[trg_ind]
        keys = []
        # template list
        keys.append(self._get_key(f'{i} tag_src', tags[src_ind]))
        keys.append(self._get_key(f'tag_src', tags[src_ind]))
        keys.append(self._get_key(f'{j} tag_trg', tags[trg_ind]))
        keys.append(self._get_key(f'tag_trg', tags[trg_ind]))
        #
        return keys

    def _get_key(self, name, *args):
        return ' '.join((name,) + tuple(args))

    def _add_keys(self, keys):
        for key in keys:
            self.features[key] += 1

    def _check_keys(self, keys):
        exist = []
        for key in keys:
            if self.features[key] != 0:
                exist.append(key)
        return exist


#####


# features to add
# TODO: add each template validates his input
# add('src suffix3', src_word[-3:])  # add len_from_end
# add('src pref1', src_word[0])
# add(f'{i} tag_src', tags[src_ind])
# add(f'{j} tag_trg', tags[trg_ind])

# add(f'{src_ind} pref2', src_word[0:1])
# add('i-2 tag_trg', tags[trg_ind])
# add('i tag+i-2 tag', prev, prev2)
# add('i word', context[i])
# add('i-1 tag+i word', prev, context[i])
# add('i-1 word', context[i - 1])
# add('i-1 suffix', context[i - 1][-3:])
# add('i-2 word', context[i - 2])
# add('i+1 word', context[i + 1])
# add('i+1 suffix', context[i + 1][-3:])
# add('i+2 word', context[i + 2])

#######

class BootCamp:

    def __init__(self, features):
        self.features = features

    def train_soldiers(self, soldier_list):
        """
        Create feature tensor for each object
        :param soldier_list:
        :type soldier_list:
        :return:
        :rtype:
        """
        print("train_soldiers not working yet, need more money")
        # TODO: decide if (1): each feature is applied to all edges, or (2): apply all features to each edge
        # TODO: (1) in this implementation feature is a function that gets a list of tuples and returns a list of edges (i,j) that activate the feature
        #  pros: ; cons:
        # TODO: (2) in this implementation feature is a function that gets a tuple and returns True\False
        # pros: ; cons:
        return soldier_list
