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

# imports
from HW2.models.dp_model import DP_sentence
from collections import defaultdict
from heapq import nlargest
from scipy import sparse as spar
import numpy as np

#

# TODO: feature definition:
# TODO remember we don't have output value restriction (can be a real number)

suffix_list_base = ['acy', 'al', 'ance', 'ence', 'dom', 'er', 'or', 'ism', 'ist', 'ity', 'ty', 'ment', 'ness', 'ship',
                    'sion', 'tion']
suffix_list_verbs = ['ate', 'en', 'ify', 'fy', 'ise', 'ize']
suffix_list_adj = ['able', 'ible', 'al', 'esque', 'ful', 'ic', 'ical', 'ious', 'ous', 'ish', 'ive', 'less', 'y']
suffix_list_adverbs = ['ly', 'ward', 'wards', 'wise']
all_suffix = suffix_list_base + suffix_list_adj + suffix_list_adverbs + suffix_list_verbs
prefix_list = ['ante', 'ante', 'circum', 'co', 'de', 'dis', 'em', 'en', 'epi', 'ex', 'extra', 'fore', 'homo', 'hype',
               'il', 'im', 'in', 'ir', 'im', 'in', 'infra', 'intra', 'inter', 'macro', 'micro', 'mid', 'mis', 'mono',
               'non', 'omni', 'para', 'post', 'pre', 're', 'ag', 'semi', 'sub', 'super', 'therm', 'trans', 'tri', 'un',
               'no', 'uni']
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
              'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
              'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these',
              'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
              'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
              'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before',
              'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
              'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
              'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
              'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']
stop_dict = defaultdict(bool)
for word in stop_words:
    stop_dict[word] = True

# TODO: dict for each

class Features:
    def __init__(self, model_type='base'):
        self.model_type = model_type
        self.features = defaultdict(int)
        self.features['bias'] = 1
        self.num_features = 1
        self.key2token = dict()

    def extract_features(self, data_obj: DP_sentence):
        """
        Creates keys in dict, a key corresponds to a feature
        The key holds the count that the feature was seen
        :param data_obj:
        :type data_obj: DP_sentence
        :return:
        :rtype:
        """
        graph = data_obj.graph_tag
        context = data_obj.sentence
        tags = data_obj.tags
        for src_ind, trg_inds in graph.items():
            for trg_ind in trg_inds:  # edge in the graph (src_ind, trg_ind)
                # current
                keys = self.get_keys(src_ind, trg_ind, context, tags, graph)
                self._add_keys(keys)

    def tokenize(self):
        self.key2token = {key: ind for ind, key in enumerate(self.features.keys())}
        self.num_features = len(list(self.features.keys()))

    def truncate_features(self, n: int):
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

    def truncate_by_thresh(self, thresh):
        temp_dict = defaultdict(int)
        keys2keep = {key: val for key, val in self.features.items() if val > thresh}
        for key in keys2keep:
            temp_dict[key] = self.features[key]
        self.features = temp_dict
        self.key2token = {key: ind for ind, key in enumerate(self.features.keys())}
        self.num_features = len(list(self.features.keys()))

    def fill_tensor(self, data_obj: DP_sentence, fast: bool = True):
        context = data_obj.sentence
        tags = data_obj.tags
        num_nodes = len(tags)
        data_obj.f = []
        if fast:
            data_obj.full_graph = self.create_init_graph(data_obj)
        else:
            data_obj.full_graph = {src: list(range(1, num_nodes)) for src in
                                   range(num_nodes)}  # TODO check with alex if this is correct
        for src_ind, trg_inds in data_obj.full_graph.items():
            rows, cols, data, indptr = [], [], [], [0]
            for trg_ind in range(0, num_nodes):
                next_ptr = indptr[-1]
                if trg_ind in trg_inds:  # edge in the graph (src_ind, trg_ind)
                    keys = self.get_keys(src_ind, trg_ind, context, tags, data_obj.full_graph)  # Todo check this part
                    exist = self._check_keys(keys)
                    activ_feat_inds = [self.key2token[activ] for activ in exist]
                    for activ_ind in activ_feat_inds:
                        rows.append(trg_ind)
                        cols.append(activ_ind)
                        data.append(True)
                    next_ptr = len(activ_feat_inds) + next_ptr
                indptr.append(next_ptr)
            indptr = np.array(indptr)
            data_obj.f.append(spar.csr_matrix((data, cols, indptr), shape=(num_nodes, self.num_features), dtype=bool))

    def get_keys(self, src_ind, trg_ind, context, tags, graph):
        src_word = context[src_ind]
        src_tag = tags[src_ind]
        trg_word = context[trg_ind]
        trg_tag = tags[trg_ind]
        keys = []
        if self.model_type == 'base':
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
            return keys

        # Bohnet (2010) Features

        valid_prev_f = lambda ind: ind > 0
        valid_next_f = lambda ind: ind < len(context) - 1
        args_dict = {}
        h_ind = trg_ind
        d_ind = src_ind
        c_ind = self.get_child(h_ind, d_ind, graph)  # child or sibling
        b_ind = int(min(src_ind, trg_ind) + (src_ind - trg_ind) / 2)

        ## head values
        w_h = context[h_ind]
        p_h = tags[h_ind]
        args_dict['wh'] = {'valid': True, 'value': w_h}
        args_dict['ph'] = {'valid': True, 'value': p_h}
        valid_prev_h = valid_prev_f(h_ind)
        args_dict['wh-1'] = {'valid': valid_prev_h}
        args_dict['ph-1'] = {'valid': valid_prev_h}
        if valid_prev_h:
            w_h_prev = context[h_ind - 1]
            p_h_prev = tags[h_ind - 1]
            args_dict['ph-1']['value'] = p_h_prev
            args_dict['wh-1']['value'] = w_h_prev
        valid_next_h = valid_next_f(h_ind)
        args_dict['wh+1'] = {'valid': valid_next_h}
        args_dict['ph+1'] = {'valid': valid_next_h}
        if valid_next_h:
            w_h_next = context[h_ind + 1]
            p_h_next = tags[h_ind + 1]
            args_dict['wh+1']['value'] = p_h_next
            args_dict['ph+1']['value'] = w_h_next

        ## dependent values
        w_d = context[d_ind]
        p_d = tags[d_ind]
        args_dict['wd'] = {'valid': True, 'value': w_d}
        args_dict['pd'] = {'valid': True, 'value': p_d}
        valid_prev_d = valid_prev_f(d_ind)
        args_dict['wd-1'] = {'valid': valid_prev_d}
        args_dict['pd-1'] = {'valid': valid_prev_d}
        if valid_prev_d:
            w_d_prev = context[d_ind - 1]
            p_d_prev = tags[d_ind - 1]
            args_dict['pd-1']['value'] = p_d_prev
            args_dict['wd-1']['value'] = w_d_prev
        valid_next_d = valid_next_f(d_ind)
        args_dict['wd+1'] = {'valid': valid_next_d}
        args_dict['pd+1'] = {'valid': valid_next_d}
        if valid_next_d:
            w_d_next = context[d_ind + 1]
            p_d_next = tags[d_ind + 1]
            args_dict['wd+1']['value'] = p_d_next
            args_dict['pd+1']['value'] = w_d_next

        d_h_d = 'L' if d_ind > h_ind else 'R'
        args_dict['d(hd)'] = {'valid': True, 'value': d_h_d}

        ## between values
        valid_b = np.abs(src_ind - trg_ind) > 1
        args_dict['wb'] = {'valid': valid_b}
        args_dict['pb'] = {'valid': valid_b}
        args_dict['wb-1'] = {'valid': valid_b}
        args_dict['pb-1'] = {'valid': valid_b}
        args_dict['wb+1'] = {'valid': valid_b}
        args_dict['pb+1'] = {'valid': valid_b}

        if valid_b:
            w_b = context[b_ind]
            p_b = tags[b_ind]
            args_dict['wb'] = {'valid': True, 'value': w_b}
            args_dict['pb'] = {'valid': True, 'value': p_b}
            valid_prev_b = valid_prev_f(b_ind)
            args_dict['wb-1'] = {'valid': valid_prev_b}
            args_dict['pb-1'] = {'valid': valid_prev_b}
            if valid_prev_b:
                w_b_prev = context[b_ind - 1]
                p_b_prev = tags[b_ind - 1]
                args_dict['pb-1']['value'] = p_b_prev
                args_dict['wb-1']['value'] = w_b_prev
            valid_next_b = valid_next_f(b_ind)
            args_dict['wb+1'] = {'valid': valid_next_b}
            args_dict['pb+1'] = {'valid': valid_next_b}
            if valid_next_b:
                w_b_next = context[b_ind + 1]
                p_b_next = tags[b_ind + 1]
                args_dict['wb+1']['value'] = p_b_next
                args_dict['pb+1']['value'] = w_b_next

        ## child values
        valid_c = c_ind != []
        args_dict['wc'] = {'valid': valid_c}
        args_dict['pc'] = {'valid': valid_c}
        args_dict['wc-1'] = {'valid': valid_c}
        args_dict['pc-1'] = {'valid': valid_c}
        args_dict['wc+1'] = {'valid': valid_c}
        args_dict['pc+1'] = {'valid': valid_c}

        if valid_c:
            w_c = context[c_ind]
            p_c = tags[c_ind]
            d_d_c = 'L' if h_ind > c_ind else 'R'
            args_dict['d(hdc)'] = {'valid': True, 'value': d_h_d + d_d_c}
            args_dict['wc'] = {'valid': True, 'value': w_c}
            args_dict['pc'] = {'valid': True, 'value': p_c}
            valid_prev_c = valid_prev_f(c_ind)
            args_dict['wc-1'] = {'valid': valid_prev_c}
            args_dict['pc-1'] = {'valid': valid_prev_c}
            if valid_prev_c:
                w_c_prev = context[c_ind - 1]
                p_c_prev = tags[c_ind - 1]
                args_dict['pc-1']['value'] = p_c_prev
                args_dict['wc-1']['value'] = w_c_prev
            valid_next_c = valid_next_f(c_ind)
            args_dict['wc+1'] = {'valid': valid_next_c}
            args_dict['pc+1'] = {'valid': valid_next_c}
            if valid_next_c:
                w_c_next = context[c_ind + 1]
                p_c_next = tags[c_ind + 1]
                args_dict['wc+1']['value'] = p_c_next
                args_dict['pc+1']['value'] = w_c_next

        # first order
        self.add_from_temp(keys, f'[wp]h,[wp]d,d(hd)', args_dict)
        self.add_from_temp(keys, f'[wp]h,d(hd)', args_dict)
        self.add_from_temp(keys, f'wd, pd, d(hd)', args_dict)
        self.add_from_temp(keys, f'[wp]d, d(hd)', args_dict)
        self.add_from_temp(keys, f'wh, ph, wd, pd, d(hd)', args_dict)
        self.add_from_temp(keys, f'ph, wh, pd, d(hd)', args_dict)
        self.add_from_temp(keys, f'wh, wd, pd, d(hd)', args_dict)
        self.add_from_temp(keys, f'wh, ph, [wp]d, d(hd)', args_dict)
        self.add_from_temp(keys, f'wh, ph, [wp]d, d(hd)', args_dict)
        self.add_from_temp(keys, f'ph, pb, pd, d(hd)', args_dict)
        self.add_from_temp(keys, f'ph, ph+1, pd-1, pd, d(hd)', args_dict)
        self.add_from_temp(keys, f'ph-1, ph, pd-1, pd, d(hd)', args_dict)
        self.add_from_temp(keys, f'ph, ph+1, pd, pd+1, d(hd)', args_dict)
        self.add_from_temp(keys, f'ph-1, ph, pd, pd+1, d(hd)', args_dict)

        if stop_dict[w_h] or stop_dict[w_d]:
            return keys
        # second order
        self.add_from_temp(keys, f'ph, pd, pc, d(hdc)', args_dict)
        self.add_from_temp(keys, f'wh, wd, wc, d(hdc)', args_dict)
        self.add_from_temp(keys, f'ph, [wp]c, d(hdc)', args_dict)
        self.add_from_temp(keys, f'pd, [wp]c, d(hdc)', args_dict)
        self.add_from_temp(keys, f'wh, [wp]c, d(hdc)', args_dict)
        self.add_from_temp(keys, f'wd, [wp]c, d(hdc)', args_dict)
        self.add_from_temp(keys, f'[wp]h, [wp]h+1, [wp]c, d(hdc)', args_dict)
        self.add_from_temp(keys, f'[wp]h-1, [wp]h, [wp]c, d(hdc)', args_dict)
        self.add_from_temp(keys, f'[wp]h, [wp]c-1, [wp]c, d(hdc)', args_dict)
        self.add_from_temp(keys, f'[wp]h, [wp]c, [wp]c+1, d(hdc)', args_dict)
        self.add_from_temp(keys, f'[wp]h-1, [wp]h, [wp]c-1, [wp]c, d(hdc)', args_dict)
        self.add_from_temp(keys, f'[wp]h-1, [wp]h, [wp]c-1, [wp]c, d(hdc)', args_dict)
        self.add_from_temp(keys, f'[wp]h-1, [wp]h, [wp]c-1, [wp]c, d(hdc)', args_dict)
        self.add_from_temp(keys, f'[wp]h, [wp]h+1, [wp]c-1, [wp]c, d(hdc)', args_dict)
        self.add_from_temp(keys, f'[wp]h-1, [wp]h, [wp]c, [wp]c+1, d(hdc)', args_dict)
        self.add_from_temp(keys, f'[wp]h, [wp]h+1, [wp]c, [wp]c+1, d(hdc)', args_dict)
        self.add_from_temp(keys, f'[wp]d, [wp]d+1, [wp]c, d(hdc)', args_dict)
        self.add_from_temp(keys, f'[wp]d-1, [wp]d, [wp]c, d(hdc)', args_dict)
        self.add_from_temp(keys, f'[wp]d, [wp]c-1, [wp]c, d(hdc)', args_dict)
        self.add_from_temp(keys, f'[wp]d, [wp]c, [wp]c+1, d(hdc)', args_dict)
        self.add_from_temp(keys, f'[wp]d, [wp]d+1, [wp]c-1, [wp]c, d(hdc)', args_dict)
        self.add_from_temp(keys, f'[wp]d, [wp]d+1, [wp]c, [wp]c+1, d(hdc)', args_dict)
        self.add_from_temp(keys, f'[wp]d-1, [wp]d, [wp]c-1, [wp]c, d(hdc)', args_dict)
        self.add_from_temp(keys, f'[wp]d-1, [wp]d, [wp]c, [wp]c+1, d(hdc)', args_dict)

        # extended feature list

        # distance between
        # self._add_key(keys, abs(trg_ind - src_ind) > 4, f'far_tag_src_tag_trg', src_tag, trg_tag)
        # self._add_key(keys, abs(trg_ind - src_ind) == 1, f'neighbour_tag_src_tag_trg', src_tag, trg_tag)
        #
        # # stop words
        # src_is_stop = 0
        # if src_word in stop_words:
        #     self._add_key(keys, True, f'stop_word_tag_src_tag_trg', src_word, src_tag, trg_tag)
        #     src_is_stop = 1
        # if trg_word in stop_words:
        #     self._add_key(keys, True, f'tag_src_stop_word_tag_trg', src_tag, trg_word, trg_tag)
        #     return keys
        # if src_is_stop:
        #     return keys
        #
        # # suffix features
        # suffix_checker = lambda src_word, suffix: len(src_word) > len(suffix) and src_word[-len(suffix):] == suffix
        # # base suffix
        # [self._add_key(keys, True, f'suffix_base_src', src_word) for suffix in suffix_list_base if
        #  suffix_checker(src_word, suffix)]
        # [self._add_key(keys, True, f'suffix_base_trg', trg_word) for suffix in suffix_list_base if
        #  suffix_checker(trg_word, suffix)]
        # [self._add_key(keys, True, f'suffix_base_src_word_trg', src_word, trg_word) for suffix in suffix_list_base if
        #  suffix_checker(src_word, suffix)]
        # [self._add_key(keys, True, f'suffix_base_src_tag_trg', src_word, trg_tag) for suffix in suffix_list_base if
        #  suffix_checker(trg_word, suffix)]
        # # verb suffix
        # [self._add_key(keys, True, f'suffix_verb_src', src_word) for suffix in suffix_list_verbs if
        #  suffix_checker(src_word, suffix)]
        # [self._add_key(keys, True, f'suffix_verb_trg', trg_word) for suffix in suffix_list_verbs if
        #  suffix_checker(trg_word, suffix)]
        # [self._add_key(keys, True, f'suffix_verb_src_word_trg', src_word, trg_word) for suffix in suffix_list_verbs if
        #  suffix_checker(src_word, suffix)]
        # [self._add_key(keys, True, f'suffix_verb_src_tag_trg', src_word, trg_tag) for suffix in suffix_list_verbs if
        #  suffix_checker(trg_word, suffix)]
        # # adj suffix
        # [self._add_key(keys, True, f'suffix_adj_src', src_word) for suffix in suffix_list_adj if
        #  suffix_checker(src_word, suffix)]
        # [self._add_key(keys, True, f'suffix_adj_trg', trg_word) for suffix in suffix_list_adj if
        #  suffix_checker(trg_word, suffix)]
        # [self._add_key(keys, True, f'suffix_adj_src_word_trg', src_word, trg_word) for suffix in suffix_list_adj if
        #  suffix_checker(src_word, suffix)]
        # [self._add_key(keys, True, f'suffix_adj_src_tag_trg', src_word, trg_tag) for suffix in suffix_list_adj if
        #  suffix_checker(trg_word, suffix)]
        # # adverb suffix
        # [self._add_key(keys, True, f'suffix_adverb_src', src_word) for suffix in suffix_list_adverbs if
        #  suffix_checker(src_word, suffix)]
        # [self._add_key(keys, True, f'suffix_adverb_trg', trg_word) for suffix in suffix_list_adverbs if
        #  suffix_checker(trg_word, suffix)]
        # [self._add_key(keys, True, f'suffix_adverb_src_word_trg', src_word, trg_word) for suffix in suffix_list_adverbs
        #  if suffix_checker(src_word, suffix)]
        # [self._add_key(keys, True, f'suffix_adverb_src_tag_trg', src_word, trg_tag) for suffix in suffix_list_adverbs if
        #  suffix_checker(trg_word, suffix)]
        #
        # # prefix templates
        # prefix_checker = lambda src_word, prefix: len(src_word) > len(prefix) and src_word[
        #                                                                           :len(prefix)].lower() == prefix
        # [self._add_key(keys, True, f'prefix_tag_src', prefix, src_word) for prefix in prefix_list if
        #  prefix_checker(src_word, prefix)]
        # [self._add_key(keys, True, f'prefix_tag_trg', prefix, trg_word) for prefix in prefix_list if
        #  prefix_checker(trg_word, prefix)]
        # [self._add_key(keys, True, f'prefix_src_word_trg', prefix, src_word, trg_word) for prefix in prefix_list
        #  if prefix_checker(src_word, prefix)]
        # [self._add_key(keys, True, f'prefix_src_tag_trg', prefix, src_word, trg_tag) for prefix in prefix_list if
        #  prefix_checker(trg_word, prefix)]
        #
        # # left-right
        # self._add_key(keys, src_ind > trg_ind, f'left_tag_src_tag_trg', src_tag, trg_tag)
        # self._add_key(keys, src_ind < trg_ind, f'right_tag_src_tag_trg', src_tag, trg_tag)
        #
        # # capital letters
        # # first letter
        # self._add_key(keys, (src_word[0].isupper() and src_ind != 1), f'upper_tag_src', src_tag)
        # self._add_key(keys, src_word[0].isupper() and src_ind != 1, f'upper_tag_src_tag_trg', src_tag, trg_tag)
        #
        # # contains digit
        # self._add_key(keys, any(c.isdigit() for c in src_word), f'contains_digit_tag_src_tag_trg', src_tag, trg_tag)
        # self._add_key(keys, any(c.isdigit() for c in trg_word), f'tag_src_contains_digit_tag_trg', src_tag, trg_tag)
        #
        # # grandchild, sibling
        #
        # # context
        # self._add_key(keys, src_ind > 0, f'word[-1]_word_src', context[src_ind - 1], src_word)
        # self._add_key(keys, src_ind > 0, f'tag[-1]_word_src', tags[src_ind - 1], trg_word)

        return keys

    def create_init_graph(self, obj):
        full_graph = {}
        sentence_len = len(obj.sentence)
        # debug_count = 0  # TODO: remove after debug
        for src in range(sentence_len):
            full_graph[src] = []
            for trg in range(sentence_len):
                if self.features[self.get_key(f'tag_src_tag_trg', obj.tags[src], obj.tags[trg])]:
                    full_graph[src].append(trg)  # TODO: save in dictionary
                    # debug_count += 1  #TODO: remove after debug
        # print(f"Created {debug_count} edges instead of {sentence_len*sentence_len}")  #TODO: remove after debug
        return full_graph

    def add_from_temp(self, keys, sig, args_dict, acc_args=[], acc_valid=True, acc_name=''):
        # example sig: '[wp]h-1, [wp]h, pc-1, [wp]c, d(h, d, c)'
        # args_dict: args_dict[w_h]-> {[valid]->True [value]->5}
        if not acc_valid:
            return
        parts = [part.replace(' ', '') for part in sig.split(',')]  # split and remove spaces
        num_parts = len(parts)
        if num_parts == 0 or len(parts[0]) == 0:
            # self._add_key(keys, True, acc_name, *acc_args)
            keys.append(self.get_key(acc_name, *acc_args))
            return
        current = parts[0]
        if current[0] == '[':
            next_typ = current[4:]
            p_part = 'p' + next_typ + ',' + ','.join(parts[1:])
            w_part = 'w' + next_typ + ',' + ','.join(parts[1:])
            self.add_from_temp(keys, p_part, args_dict, acc_args, acc_valid, acc_name)
            self.add_from_temp(keys, w_part, args_dict, acc_args, acc_valid, acc_name)
        else:
            acc_valid_ = args_dict[current]['valid'] and acc_valid
            if not acc_valid_:
                return
            acc_args_ = [args_dict[current]['value']] + acc_args
            acc_name_ = acc_name + current
            self.add_from_temp(keys, ','.join(parts[1:]), args_dict, acc_args_, acc_valid_, acc_name=acc_name_)

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

    def get_child(self, h_ind, d_ind, graph):
        if h_ind in graph and graph[h_ind] != []:
            return graph[h_ind][0]
        elif len(graph[d_ind]) > 1:
            for sibling in graph[d_ind]:
                if sibling != h_ind:
                    return sibling
        else:
            return []


class BootCamp:

    def __init__(self, features):

        assert isinstance(features, Features)
        self.features = features

    def investigate_soldiers(self, soldier_list):
        for soldier in soldier_list:
            self.features.extract_features(soldier)

    def truncate_features(self, n):
        self.features.truncate_features(n)

    def train_soldiers(self, soldier_list, fast=True):
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
            self.features.fill_tensor(soldier, fast=fast)

        # return soldier_list  # inplace

    def get_model(self):
        return self.features.model_type
