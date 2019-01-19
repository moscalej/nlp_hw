# imports
from collections import defaultdict
from heapq import nlargest
from scipy import sparse as spar

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
    def __init__(self, model_type='base'):
        self.model_type = model_type
        self.features = defaultdict(int)
        self.features['bias'] = 1
        self.num_features = 1
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
                keys = self.get_keys(src_ind, trg_ind, context, tags)
                self._add_keys(keys)

    def tokenize(self):
        self.key2token = {key: ind for ind, key in enumerate(self.features.keys())}
        self.num_features = len(list(self.features.keys()))

    def truncate_features(self, n):
        """
        Truncate n most frequent features
        :param n:
        :type n:
        :return:
        :rtype:
        """
        keys2keep = nlargest(n, self.features, key=self.features.get)
        # sorted(self.features, key=self.features.get, reverse=True)
        temp_dict = defaultdict(int)
        for key in keys2keep:
            temp_dict[key] = self.features[key]
        self.features = temp_dict
        self.key2token = {key: ind for ind, key in enumerate(self.features.keys())}
        self.num_features = len(list(self.features.keys()))

    def fill_tensor(self, data_obj):
        context = data_obj.sentence
        tags = data_obj.tags
        num_nodes = len(tags)
        # graph = get_full_graph(num_nodes)
        # data_obj.f = [spar.csc_matrix((num_nodes, self.num_features), dtype=bool) for _ in range(num_nodes)]
        data_obj.f = []
        data_obj.graph_est = self.create_init_graph(data_obj)
        for src_ind, trg_inds in data_obj.graph_est.items():
            rows, cols, data = [], [], []
            for trg_ind in trg_inds:  # edge in the graph (src_ind, trg_ind)
                # for src_ind in range(num_nodes):
                #     rows, cols, data = [], [], []
                #     for trg_ind in range(1, num_nodes):  # edge in the graph (src_ind, trg_ind)
                keys = self.get_keys(src_ind, trg_ind, context, tags)
                exist = self._check_keys(keys)
                activ_feat_inds = [self.key2token[activ] for activ in exist]
                for activ_ind in activ_feat_inds:
                    rows.append(trg_ind)
                    cols.append(activ_ind)
                    data.append(True)
            data_obj.f.append(spar.coo_matrix((data, (rows, cols)), shape=(num_nodes, self.num_features), dtype=bool))
            # data_obj.f[src_ind][trg_ind, activ_feat_inds] = True

    def get_keys(self, src_ind, trg_ind, context, tags):
        src_word = context[src_ind]
        src_tag = tags[src_ind]
        trg_word = context[trg_ind]
        trg_tag = tags[trg_ind]
        keys = []
        # basic template list
        # uni-grams:
        self._add_key(keys, True, f'word_src', src_word)
        self._add_key(keys, True, f'tag_src', src_tag)
        self._add_key(keys, True, f'word_tag_src', src_word, src_tag)
        self._add_key(keys, True, f'word_trg', trg_word)
        self._add_key(keys, True, f'tag_trg', trg_tag)
        self._add_key(keys, True, f'word_tag_trg', trg_word, trg_tag)
        # bi-grams:
        self._add_key(keys, True, f'word_tag_src_word_tag_trg', src_word, src_tag, trg_word, trg_tag)
        self._add_key(keys, True, f'tag_src_word_tag_trg', src_tag, trg_word, trg_tag)
        self._add_key(keys, True, f'word_src_word_tag_trg', src_word, trg_word, trg_tag)
        self._add_key(keys, True, f'word_tag_src_tag_trg', src_word, src_tag, trg_tag)
        self._add_key(keys, True, f'word_tag_src_word_trg', src_word, src_tag, trg_word)
        self._add_key(keys, True, f'word_src_word_trg', src_word, trg_word)
        self._add_key(keys, True, f'tag_src_tag_trg', src_tag, trg_tag)

        if self.model_type == 'base':
            return keys
        # extended template list
        # keys.append(self._get_key(f'{src_ind} tag_src', src_tag))
        self._add_key(keys, True, f'{src_ind} tag_src', src_tag)
        # keys.append(self._get_key(f'{src_ind} word_src', src_word))
        self._add_key(keys, True, f'{src_ind} tag_src', src_tag)
        # keys.append(self._get_key(f'tag_src', src_tag))
        self._add_key(keys, True, f'tag_src', src_tag)
        # keys.append(self._get_key(f'{trg_ind} tag_trg', trg_tag))
        self._add_key(keys, True, f'{trg_ind} tag_trg', trg_tag)
        # keys.append(self._get_key(f'{trg_ind} word_trg', trg_word))
        self._add_key(keys, True, f'{trg_ind} word_trg', trg_word)
        # keys.append(self._get_key(f'{src_ind} to {trg_ind}', ''))
        self._add_key(keys, True, f'{src_ind} to {trg_ind}', '')
        # if src_ind > 0:
        #     keys.append(self._get_key(f'{src_ind-1} word', context[src_ind - 1]))
        self._add_key(keys, src_ind > 0, f'{src_ind-1} word', context[src_ind - 1])
        # keys.append(self._get_key(f'tag_trg', trg_tag))
        #
        return keys

    def create_init_graph(self, obj):
        full_graph = {}
        sentence_len = len(obj.sentence)
        # debug_count = 0  #TODO: remove after debug
        for src in range(sentence_len):
            full_graph[src] = []
            for trg in range(sentence_len):
                if self.features[self.get_key(f'tag_src_tag_trg', obj.tags[src], obj.tags[trg])]:
                    full_graph[src].append(trg)  # TODO: save in dictionary
                    # debug_count += 1  #TODO: remove after debug
        # print(f"Created {debug_count} edges instead of {sentence_len*sentence_len}")  #TODO: remove after debug
        return full_graph

    def _add_key(self, key_container, valid, name, *args):
        if valid:
            key_container.append(self.get_key(name, *args))

    def get_key(self, name, *args):
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
# add(f'{j} tag_trg', trg_tag)

# add(f'{src_ind} pref2', src_word[0:1])
# add('i-2 tag_trg', trg_tag)
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

        assert isinstance(features, Features)
        self.features = features

    def investigate_soldiers(self, soldier_list):
        for soldier in soldier_list:
            self.features.extract_features(soldier)

    def truncate_features(self, n):
        self.features.truncate_features(n)

    def train_soldiers(self, soldier_list):
        """
        Create feature tensor for each object
        :param soldier_list:
        :type soldier_list:
        :return:
        :rtype:
        """
        # if no truncation has been made, generate key2token
        if len(list(self.features.key2token.keys())) == 0:
            self.features.tokenize()
        # fill tensor
        for soldier in soldier_list:
            self.features.fill_tensor(soldier)

        # return soldier_list  # inplace

    def get_model(self):
        return self.features.model_type
