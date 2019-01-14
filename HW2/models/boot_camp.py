# imports
from collections import defaultdict


#

# TODO: feature definition:
# TODO remember we don't have output value restriction (can be a real number)

class Features:
    def __init__(self):
        features = defaultdict(int)
        features['bias'] = 1

    def extract_features(self, data_obj):
        """
        Creates keys in dict, a key corresponds to a feature
        :param data_obj:
        :type data_obj:
        :return:
        :rtype:
        """
        graph = data_obj.graph_tag
        context = data_obj.sentence
        tags = data_obj.tags

        def add(name, *args):
            self.features[' '.join((name,) + tuple(args))] += 1  # TODO decide if better to use =1 instead
            # keep key to ind dict

        for src_ind, trg_inds in graph.items():
            for trg_ind in trg_inds:  # edge in the graph (src_ind, trg_ind)
                # current
                src_word = context[src_ind]
                trg_word = context[trg_ind]
                # features to add
                # TODO: add each template validates his input
                add('src suffix3', src_word[-3:])  # add len_from_end
                add('src pref1', src_word[0])
                add(f'{src_ind} tag_src', tags[src_ind])

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

    def get_obj_keys(self, data_obj):

        graph = data_obj.graph_tag  # TODO: full graph
        context = data_obj.sentence
        tags = data_obj.tags
        for src_ind, trg_inds in graph.items():
            for trg_ind in trg_inds:  # edge in the graph (src_ind, trg_ind)
                # current
                src_word = context[src_ind]
                trg_word = context[trg_ind]

                self._get_key('bias')
                self._get_key('suffix', src_word[-3:])
                self._get_key('pref1', src_word[0])
                self._get_key('i pref2', src_word[0:1])
                self._get_key('i-1 tag_src', tags[src_ind])
                self._get_key('i-2 tag_trg', tags[trg_ind])
                self._get_key('i tag+i-2 tag', prev, prev2)
                self._get_key('i word', context[i])
                self._get_key('i-1 tag+i word', prev, context[i])
                self._get_key('i-1 word', context[i - 1])
                self._get_key('i-1 suffix', context[i - 1][-3:])
                self._get_key('i-2 word', context[i - 2])
                self._get_key('i+1 word', context[i + 1])
                self._get_key('i+1 suffix', context[i + 1][-3:])
                self._get_key('i+2 word', context[i + 2])

    def _get_key(self, name, *args):
        return ' '.join((name,) + tuple(args))

    def _get_key_index(self, key):
        return ' '.join((name,) + tuple(args))

    def _set_key(self, key, dict):
        dict[key] += 1

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
