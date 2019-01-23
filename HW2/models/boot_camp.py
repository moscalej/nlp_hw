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
from tqdm import tqdm, tqdm_gui
from models.data_object import DP_sentence
from collections import defaultdict
from heapq import nlargest, nsmallest
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
        self.features_full = None
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
        self.features_full = self.features.copy()
        self.key2token = {key: ind for ind, key in enumerate(self.features.keys())}
        self.num_features = len(list(self.features.keys()))

    def truncate_features(self, remove_top: int, remove_bottom: int):
        """
        Truncate n most frequent features
        :param n:
        :type n:
        :return:
        :rtype:
        """
        num_features = len(list(self.features.keys()))

        n_top = num_features - remove_bottom
        n_bottom = num_features - remove_top
        keys2keep_top = set(nlargest(n_top, self.features, key=self.features.get))
        if n_bottom:
            keys2keep_bottom = set(nsmallest(n_bottom, self.features, key=self.features.get))
            keys2keep = list(keys2keep_top.intersection(keys2keep_bottom))
        else:
            keys2keep = keys2keep_top

        # sorted(self.features, key=self.features.get, reverse=True)
        temp_dict = defaultdict(int)
        for key in keys2keep:
            temp_dict[key] = self.features[key]
        self.features_full = self.features.copy()
        self.features = temp_dict
        self.key2token = {key: ind for ind, key in enumerate(self.features.keys())}
        self.num_features = len(list(self.features.keys()))

    def truncate_by_thresh(self, n_top, n_bottom):
        temp_dict = defaultdict(int)
        keys2keep = {key: val for key, val in self.features.items() if val < n_top and val > n_bottom}
        for key in keys2keep:
            temp_dict[key] = self.features[key]
        self.features_full = self.features.copy()
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
        c_inds = self.get_child_list(h_ind, d_ind, graph)  # child or sibling
        b_ind = int(min(src_ind, trg_ind) + (src_ind - trg_ind) / 2)

        ## head values
        w_h = []
        p_h = []
        w_h_prev = []
        p_h_prev = []
        w_h_next = []
        p_h_next = []
        valid_next_h = []
        valid_prev_h = []
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
        w_d = []
        p_d = []
        w_d_prev = []
        p_d_prev = []
        w_d_next = []
        p_d_next = []
        valid_next_d = []
        valid_prev_d = []
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
        w_b = []
        p_b = []
        w_b_prev = []
        p_b_prev = []
        w_b_next = []
        p_b_next = []
        valid_next_b = []
        valid_prev_b = []
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

        # first order
        # self.add_from_temp(keys, f'[wp]h,[wp]d,d(hd)', args_dict)
        # self.add_from_temp(keys, f'[wp]h,d(hd)', args_dict)
        # self.add_from_temp(keys, f'wd, pd, d(hd)', args_dict)
        # self.add_from_temp(keys, f'[wp]d, d(hd)', args_dict)
        # self.add_from_temp(keys, f'wh, ph, wd, pd, d(hd)', args_dict)
        # self.add_from_temp(keys, f'ph, wh, pd, d(hd)', args_dict)
        # self.add_from_temp(keys, f'wh, wd, pd, d(hd)', args_dict)
        # self.add_from_temp(keys, f'wh, ph, [wp]d, d(hd)', args_dict)
        # self.add_from_temp(keys, f'wh, ph, [wp]d, d(hd)', args_dict)
        # self.add_from_temp(keys, f'ph, pb, pd, d(hd)', args_dict)
        # self.add_from_temp(keys, f'ph, ph+1, pd-1, pd, d(hd)', args_dict)
        # self.add_from_temp(keys, f'ph-1, ph, pd-1, pd, d(hd)', args_dict)
        # self.add_from_temp(keys, f'ph, ph+1, pd, pd+1, d(hd)', args_dict)
        # self.add_from_temp(keys, f'ph-1, ph, pd, pd+1, d(hd)', args_dict)
        self._add_key(keys, True, 'phpdd(hd),', p_h, p_d, d_h_d)
        self._add_key(keys, True, 'phwdd(hd),', p_h, w_d, d_h_d)
        self._add_key(keys, True, 'whpdd(hd),', w_h, p_d, d_h_d)
        self._add_key(keys, True, 'whwdd(hd),', w_h, w_d, d_h_d)
        self._add_key(keys, True, 'phd(hd),', p_h, d_h_d)
        self._add_key(keys, True, 'whd(hd),', w_h, d_h_d)
        self._add_key(keys, True, 'wdpdd(hd),', w_d, p_d, d_h_d)
        self._add_key(keys, True, 'pdd(hd),', p_d, d_h_d)
        self._add_key(keys, True, 'wdd(hd),', w_d, d_h_d)
        self._add_key(keys, True, 'whphwdpdd(hd),', w_h, p_h, w_d, p_d,
                      d_h_d)
        self._add_key(keys, True, 'phwhpdd(hd),', p_h, w_h, p_d, d_h_d)
        self._add_key(keys, True, 'whwdpdd(hd),', w_h, w_d, p_d, d_h_d)
        self._add_key(keys, True, 'whphpdd(hd),', w_h, p_h, p_d, d_h_d)
        self._add_key(keys, True, 'whphwdd(hd),', w_h, p_h, w_d, d_h_d)
        self._add_key(keys, True, 'whphpdd(hd),', w_h, p_h, p_d, d_h_d)
        self._add_key(keys, True, 'whphwdd(hd),', w_h, p_h, w_d, d_h_d)
        self._add_key(keys, valid_b, 'phpbpdd(hd),', p_h, p_b, p_d, d_h_d)
        self._add_key(keys, valid_next_h and valid_prev_d, 'phph+1pd-1pdd(hd),',
                      p_h, p_h_next, p_d_prev, p_d, d_h_d)
        self._add_key(keys, valid_prev_h and valid_prev_d, 'ph-1phpd-1pdd(hd),',
                      p_h_prev, p_h, p_d_prev, p_d, d_h_d)
        self._add_key(keys, valid_next_h and valid_next_d, 'phph+1pdpd+1d(hd),',
                      p_h, p_h_next, p_d, p_d_next, d_h_d)
        self._add_key(keys, valid_prev_h and valid_next_d, 'ph-1phpdpd+1d(hd),',
                      p_h_prev, p_h, p_d, p_d_next, d_h_d)

        if stop_dict[w_h] or stop_dict[w_d]:
            return keys
        # second order
        ## child values
        w_c = []
        p_c = []
        w_c_prev = []
        p_c_prev = []
        w_c_next = []
        p_c_next = []
        valid_next_c = []
        valid_prev_c = []
        d_d_c = []
        valid_c = c_inds != []
        args_dict['wc'] = {'valid': valid_c}
        args_dict['pc'] = {'valid': valid_c}
        args_dict['wc-1'] = {'valid': valid_c}
        args_dict['pc-1'] = {'valid': valid_c}
        args_dict['wc+1'] = {'valid': valid_c}
        args_dict['pc+1'] = {'valid': valid_c}
        args_dict['d(hdc)'] = {'valid': valid_c}
        if valid_c:
            for c_ind in c_inds:
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

                self._add_key(keys, valid_c and valid_next_c and valid_c, 'phpcpc+1d(hdc),', p_h, p_c,
                              p_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_c and valid_next_c and valid_c, 'phpcwc+1d(hdc),', p_h, p_c,
                              w_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_c and valid_next_c and valid_c, 'phwcpc+1d(hdc),', p_h, w_c,
                              p_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_c and valid_next_c and valid_c, 'phwcwc+1d(hdc),', p_h, w_c,
                              w_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_c and valid_next_c and valid_c, 'whpcpc+1d(hdc),', w_h, p_c,
                              p_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_c and valid_next_c and valid_c, 'whpcwc+1d(hdc),', w_h, p_c,
                              w_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_c and valid_next_c and valid_c, 'whwcpc+1d(hdc),', w_h, w_c,
                              p_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_c and valid_next_c and valid_c, 'whwcwc+1d(hdc),', w_h, w_c,
                              w_c_next, d_h_d + d_d_c)

                self._add_key(keys, valid_c and valid_c, 'phpdpcd(hdc),', p_h, p_d, p_c,
                              d_h_d + d_d_c)
                self._add_key(keys, valid_c and valid_c, 'whwdwcd(hdc),', w_h, w_d, w_c,
                              d_h_d + d_d_c)
                self._add_key(keys, valid_c and valid_c, 'phpcd(hdc),', p_h, p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_c and valid_c, 'phwcd(hdc),', p_h, w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_c and valid_c, 'pdpcd(hdc),', p_d, p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_c and valid_c, 'pdwcd(hdc),', p_d, w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_c and valid_c, 'whpcd(hdc),', w_h, p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_c and valid_c, 'whwcd(hdc),', w_h, w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_c and valid_c, 'wdpcd(hdc),', w_d, p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_c and valid_c, 'wdwcd(hdc),', w_d, w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_next_h and valid_c and valid_c, 'phph+1pcd(hdc),', p_h,
                              p_h_next,
                              p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_next_h and valid_c and valid_c, 'phph+1wcd(hdc),', p_h,
                              p_h_next,
                              w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_next_h and valid_c and valid_c, 'phwh+1pcd(hdc),', p_h,
                              w_h_next,
                              p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_next_h and valid_c and valid_c, 'phwh+1wcd(hdc),', p_h,
                              w_h_next,
                              w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_next_h and valid_c and valid_c, 'whph+1pcd(hdc),', w_h,
                              p_h_next,
                              p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_next_h and valid_c and valid_c, 'whph+1wcd(hdc),', w_h,
                              p_h_next,
                              w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_next_h and valid_c and valid_c, 'whwh+1pcd(hdc),', w_h,
                              w_h_next,
                              p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_next_h and valid_c and valid_c, 'whwh+1wcd(hdc),', w_h,
                              w_h_next,
                              w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_h and valid_c and valid_c, 'ph-1phpcd(hdc),', p_h_prev,
                              p_h,
                              p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_h and valid_c and valid_c, 'ph-1phwcd(hdc),', p_h_prev,
                              p_h,
                              w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_h and valid_c and valid_c, 'ph-1whpcd(hdc),', p_h_prev,
                              w_h,
                              p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_h and valid_c and valid_c, 'ph-1whwcd(hdc),', p_h_prev,
                              w_h,
                              w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_h and valid_c and valid_c, 'wh-1phpcd(hdc),', w_h_prev,
                              p_h,
                              p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_h and valid_c and valid_c, 'wh-1phwcd(hdc),', w_h_prev,
                              p_h,
                              w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_h and valid_c and valid_c, 'wh-1whpcd(hdc),', w_h_prev,
                              w_h,
                              p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_h and valid_c and valid_c, 'wh-1whwcd(hdc),', w_h_prev,
                              w_h,
                              w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_c and valid_c, 'phpc-1pcd(hdc),', p_h,
                              p_c_prev,
                              p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_c and valid_c, 'phpc-1wcd(hdc),', p_h,
                              p_c_prev,
                              w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_c and valid_c, 'phwc-1pcd(hdc),', p_h,
                              w_c_prev,
                              p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_c and valid_c, 'phwc-1wcd(hdc),', p_h,
                              w_c_prev,
                              w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_c and valid_c, 'whpc-1pcd(hdc),', w_h,
                              p_c_prev,
                              p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_c and valid_c, 'whpc-1wcd(hdc),', w_h,
                              p_c_prev,
                              w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_c and valid_c, 'whwc-1pcd(hdc),', w_h,
                              w_c_prev,
                              p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_c and valid_next_c and valid_c, 'phpcwc+1d(hdc),', p_h, p_c,
                              w_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_c and valid_c, 'whwc-1wcd(hdc),', w_h,
                              w_c_prev,
                              w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_c and valid_next_c and valid_c, 'phpcpc+1d(hdc),', p_h, p_c,
                              p_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_c and valid_next_c and valid_c, 'phwcpc+1d(hdc),', p_h, w_c,
                              p_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_c and valid_next_c and valid_c, 'phwcwc+1d(hdc),', p_h, w_c,
                              w_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_c and valid_next_c and valid_c, 'whpcpc+1d(hdc),', w_h, p_c,
                              p_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_c and valid_next_c and valid_c, 'whpcwc+1d(hdc),', w_h, p_c,
                              w_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_c and valid_next_c and valid_c, 'whwcpc+1d(hdc),', w_h, w_c,
                              p_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_c and valid_next_c and valid_c, 'whwcwc+1d(hdc),', w_h, w_c,
                              w_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_prev_h and valid_c and valid_c,
                              'ph-1phpc-1pcd(hdc),', p_h_prev, p_h, p_c_prev, p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_prev_h and valid_c and valid_c,
                              'ph-1phpc-1wcd(hdc),', p_h_prev, p_h, p_c_prev, w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_prev_h and valid_c and valid_c,
                              'ph-1phwc-1pcd(hdc),', p_h_prev, p_h, w_c_prev, p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_prev_h and valid_c and valid_c,
                              'ph-1phwc-1wcd(hdc),', p_h_prev, p_h, w_c_prev, w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_prev_h and valid_c and valid_c,
                              'ph-1whpc-1pcd(hdc),', p_h_prev, w_h, p_c_prev, p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_prev_h and valid_c and valid_c,
                              'ph-1whpc-1wcd(hdc),', p_h_prev, w_h, p_c_prev, w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_prev_h and valid_c and valid_c,
                              'ph-1whwc-1pcd(hdc),', p_h_prev, w_h, w_c_prev, p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_prev_h and valid_c and valid_c,
                              'ph-1whwc-1wcd(hdc),', p_h_prev, w_h, w_c_prev, w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_prev_h and valid_c and valid_c,
                              'wh-1phpc-1pcd(hdc),', w_h_prev, p_h, p_c_prev, p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_prev_h and valid_c and valid_c,
                              'wh-1phpc-1wcd(hdc),', w_h_prev, p_h, p_c_prev, w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_prev_h and valid_c and valid_c,
                              'wh-1phwc-1pcd(hdc),', w_h_prev, p_h, w_c_prev, p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_prev_h and valid_c and valid_c,
                              'wh-1phwc-1wcd(hdc),', w_h_prev, p_h, w_c_prev, w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_prev_h and valid_c and valid_c,
                              'wh-1whpc-1pcd(hdc),', w_h_prev, w_h, p_c_prev, p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_prev_h and valid_c and valid_c,
                              'wh-1whpc-1wcd(hdc),', w_h_prev, w_h, p_c_prev, w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_prev_h and valid_c and valid_c,
                              'wh-1whwc-1pcd(hdc),', w_h_prev, w_h, w_c_prev, p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_prev_h and valid_c and valid_c,
                              'wh-1whwc-1wcd(hdc),', w_h_prev, w_h, w_c_prev, w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_prev_h and valid_c and valid_c,
                              'ph-1phpc-1pcd(hdc),', p_h_prev, p_h, p_c_prev, p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_prev_h and valid_c and valid_c,
                              'ph-1phpc-1wcd(hdc),', p_h_prev, p_h, p_c_prev, w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_prev_h and valid_c and valid_c,
                              'ph-1phwc-1pcd(hdc),', p_h_prev, p_h, w_c_prev, p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_prev_h and valid_c and valid_c,
                              'ph-1phwc-1wcd(hdc),', p_h_prev, p_h, w_c_prev, w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_prev_h and valid_c and valid_c,
                              'ph-1whpc-1pcd(hdc),', p_h_prev, w_h, p_c_prev, p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_prev_h and valid_c and valid_c,
                              'ph-1whpc-1wcd(hdc),', p_h_prev, w_h, p_c_prev, w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_prev_h and valid_c and valid_c,
                              'ph-1whwc-1pcd(hdc),', p_h_prev, w_h, w_c_prev, p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_prev_h and valid_c and valid_c,
                              'ph-1whwc-1wcd(hdc),', p_h_prev, w_h, w_c_prev, w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_prev_h and valid_c and valid_c,
                              'wh-1phpc-1pcd(hdc),', w_h_prev, p_h, p_c_prev, p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_prev_h and valid_c and valid_c,
                              'wh-1phpc-1wcd(hdc),', w_h_prev, p_h, p_c_prev, w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_prev_h and valid_c and valid_c,
                              'wh-1phwc-1pcd(hdc),', w_h_prev, p_h, w_c_prev, p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_prev_h and valid_c and valid_c,
                              'wh-1phwc-1wcd(hdc),', w_h_prev, p_h, w_c_prev, w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_prev_h and valid_c and valid_c,
                              'wh-1whpc-1pcd(hdc),', w_h_prev, w_h, p_c_prev, p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_prev_h and valid_c and valid_c,
                              'wh-1whpc-1wcd(hdc),', w_h_prev, w_h, p_c_prev, w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_prev_h and valid_c and valid_c,
                              'wh-1whwc-1pcd(hdc),', w_h_prev, w_h, w_c_prev, p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_prev_h and valid_c and valid_c,
                              'wh-1whwc-1wcd(hdc),', w_h_prev, w_h, w_c_prev, w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_prev_h and valid_c and valid_c,
                              'ph-1phpc-1pcd(hdc),', p_h_prev, p_h, p_c_prev, p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_prev_h and valid_c and valid_c,
                              'ph-1phpc-1wcd(hdc),', p_h_prev, p_h, p_c_prev, w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_prev_h and valid_c and valid_c,
                              'ph-1phwc-1pcd(hdc),', p_h_prev, p_h, w_c_prev, p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_prev_h and valid_c and valid_c,
                              'ph-1phwc-1wcd(hdc),', p_h_prev, p_h, w_c_prev, w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_prev_h and valid_c and valid_c,
                              'ph-1whpc-1pcd(hdc),', p_h_prev, w_h, p_c_prev, p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_prev_h and valid_c and valid_c,
                              'ph-1whpc-1wcd(hdc),', p_h_prev, w_h, p_c_prev, w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_prev_h and valid_c and valid_c,
                              'ph-1whwc-1pcd(hdc),', p_h_prev, w_h, w_c_prev, p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_prev_h and valid_c and valid_c,
                              'ph-1whwc-1wcd(hdc),', p_h_prev, w_h, w_c_prev, w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_prev_h and valid_c and valid_c,
                              'wh-1phpc-1pcd(hdc),', w_h_prev, p_h, p_c_prev, p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_prev_h and valid_c and valid_c,
                              'wh-1phpc-1wcd(hdc),', w_h_prev, p_h, p_c_prev, w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_prev_h and valid_c and valid_c,
                              'wh-1phwc-1pcd(hdc),', w_h_prev, p_h, w_c_prev, p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_prev_h and valid_c and valid_c,
                              'wh-1phwc-1wcd(hdc),', w_h_prev, p_h, w_c_prev, w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_prev_h and valid_c and valid_c,
                              'wh-1whpc-1pcd(hdc),', w_h_prev, w_h, p_c_prev, p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_prev_h and valid_c and valid_c,
                              'wh-1whpc-1wcd(hdc),', w_h_prev, w_h, p_c_prev, w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_prev_h and valid_c and valid_c,
                              'wh-1whwc-1pcd(hdc),', w_h_prev, w_h, w_c_prev, p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_prev_h and valid_c and valid_c,
                              'wh-1whwc-1wcd(hdc),', w_h_prev, w_h, w_c_prev, w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_next_h and valid_prev_c and valid_c and valid_c,
                              'phph+1pc-1pcd(hdc),', p_h, p_h_next, p_c_prev, p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_next_h and valid_prev_c and valid_c and valid_c,
                              'phph+1pc-1wcd(hdc),', p_h, p_h_next, p_c_prev, w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_next_h and valid_prev_c and valid_c and valid_c,
                              'phph+1wc-1pcd(hdc),', p_h, p_h_next, w_c_prev, p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_next_h and valid_prev_c and valid_c and valid_c,
                              'phph+1wc-1wcd(hdc),', p_h, p_h_next, w_c_prev, w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_next_h and valid_prev_c and valid_c and valid_c,
                              'phwh+1pc-1pcd(hdc),', p_h, w_h_next, p_c_prev, p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_next_h and valid_prev_c and valid_c and valid_c,
                              'phwh+1pc-1wcd(hdc),', p_h, w_h_next, p_c_prev, w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_next_h and valid_prev_c and valid_c and valid_c,
                              'phwh+1wc-1pcd(hdc),', p_h, w_h_next, w_c_prev, p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_next_h and valid_prev_c and valid_c and valid_c,
                              'phwh+1wc-1wcd(hdc),', p_h, w_h_next, w_c_prev, w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_next_h and valid_prev_c and valid_c and valid_c,
                              'whph+1pc-1pcd(hdc),', w_h, p_h_next, p_c_prev, p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_next_h and valid_prev_c and valid_c and valid_c,
                              'whph+1pc-1wcd(hdc),', w_h, p_h_next, p_c_prev, w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_next_h and valid_prev_c and valid_c and valid_c,
                              'whph+1wc-1pcd(hdc),', w_h, p_h_next, w_c_prev, p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_next_h and valid_prev_c and valid_c and valid_c,
                              'whph+1wc-1wcd(hdc),', w_h, p_h_next, w_c_prev, w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_next_h and valid_prev_c and valid_c and valid_c,
                              'whwh+1pc-1pcd(hdc),', w_h, w_h_next, p_c_prev, p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_next_h and valid_prev_c and valid_c and valid_c,
                              'whwh+1pc-1wcd(hdc),', w_h, w_h_next, p_c_prev, w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_next_h and valid_prev_c and valid_c and valid_c,
                              'whwh+1wc-1pcd(hdc),', w_h, w_h_next, w_c_prev, p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_next_h and valid_prev_c and valid_c and valid_c,
                              'whwh+1wc-1wcd(hdc),', w_h, w_h_next, w_c_prev, w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_h and valid_c and valid_next_c and valid_c,
                              'ph-1phpcpc+1d(hdc),', p_h_prev, p_h, p_c, p_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_h and valid_c and valid_next_c and valid_c,
                              'ph-1phpcwc+1d(hdc),', p_h_prev, p_h, p_c, w_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_h and valid_c and valid_next_c and valid_c,
                              'ph-1phwcpc+1d(hdc),', p_h_prev, p_h, w_c, p_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_h and valid_c and valid_next_c and valid_c,
                              'ph-1phwcwc+1d(hdc),', p_h_prev, p_h, w_c, w_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_h and valid_c and valid_next_c and valid_c,
                              'ph-1whpcpc+1d(hdc),', p_h_prev, w_h, p_c, p_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_h and valid_c and valid_next_c and valid_c,
                              'ph-1whpcwc+1d(hdc),', p_h_prev, w_h, p_c, w_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_h and valid_c and valid_next_c and valid_c,
                              'ph-1whwcpc+1d(hdc),', p_h_prev, w_h, w_c, p_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_h and valid_c and valid_next_c and valid_c,
                              'ph-1whwcwc+1d(hdc),', p_h_prev, w_h, w_c, w_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_h and valid_c and valid_next_c and valid_c,
                              'wh-1phpcpc+1d(hdc),', w_h_prev, p_h, p_c, p_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_h and valid_c and valid_next_c and valid_c,
                              'wh-1phpcwc+1d(hdc),', w_h_prev, p_h, p_c, w_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_h and valid_c and valid_next_c and valid_c,
                              'wh-1phwcpc+1d(hdc),', w_h_prev, p_h, w_c, p_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_h and valid_c and valid_next_c and valid_c,
                              'wh-1phwcwc+1d(hdc),', w_h_prev, p_h, w_c, w_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_h and valid_c and valid_next_c and valid_c,
                              'wh-1whpcpc+1d(hdc),', w_h_prev, w_h, p_c, p_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_h and valid_c and valid_next_c and valid_c,
                              'wh-1whpcwc+1d(hdc),', w_h_prev, w_h, p_c, w_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_h and valid_c and valid_next_c and valid_c,
                              'wh-1whwcpc+1d(hdc),', w_h_prev, w_h, w_c, p_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_h and valid_c and valid_next_c and valid_c,
                              'wh-1whwcwc+1d(hdc),', w_h_prev, w_h, w_c, w_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_next_h and valid_c and valid_next_c and valid_c,
                              'phph+1pcpc+1d(hdc),', p_h, p_h_next, p_c, p_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_next_h and valid_c and valid_next_c and valid_c,
                              'phph+1pcwc+1d(hdc),', p_h, p_h_next, p_c, w_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_next_h and valid_c and valid_next_c and valid_c,
                              'phph+1wcpc+1d(hdc),', p_h, p_h_next, w_c, p_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_next_h and valid_c and valid_next_c and valid_c,
                              'phph+1wcwc+1d(hdc),', p_h, p_h_next, w_c, w_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_next_h and valid_c and valid_next_c and valid_c,
                              'phwh+1pcpc+1d(hdc),', p_h, w_h_next, p_c, p_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_next_h and valid_c and valid_next_c and valid_c,
                              'phwh+1pcwc+1d(hdc),', p_h, w_h_next, p_c, w_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_next_h and valid_c and valid_next_c and valid_c,
                              'phwh+1wcpc+1d(hdc),', p_h, w_h_next, w_c, p_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_next_h and valid_c and valid_next_c and valid_c,
                              'phwh+1wcwc+1d(hdc),', p_h, w_h_next, w_c, w_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_next_h and valid_c and valid_next_c and valid_c,
                              'whph+1pcpc+1d(hdc),', w_h, p_h_next, p_c, p_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_next_h and valid_c and valid_next_c and valid_c,
                              'whph+1pcwc+1d(hdc),', w_h, p_h_next, p_c, w_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_next_h and valid_c and valid_next_c and valid_c,
                              'whph+1wcpc+1d(hdc),', w_h, p_h_next, w_c, p_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_next_h and valid_c and valid_next_c and valid_c,
                              'whph+1wcwc+1d(hdc),', w_h, p_h_next, w_c, w_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_next_h and valid_c and valid_next_c and valid_c,
                              'whwh+1pcpc+1d(hdc),', w_h, w_h_next, p_c, p_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_next_h and valid_c and valid_next_c and valid_c,
                              'whwh+1pcwc+1d(hdc),', w_h, w_h_next, p_c, w_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_next_h and valid_c and valid_next_c and valid_c,
                              'whwh+1wcpc+1d(hdc),', w_h, w_h_next, w_c, p_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_next_h and valid_c and valid_next_c and valid_c,
                              'whwh+1wcwc+1d(hdc),', w_h, w_h_next, w_c, w_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_next_d and valid_c and valid_c, 'pdpd+1pcd(hdc),', p_d,
                              p_d_next,
                              p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_next_d and valid_c and valid_c, 'pdpd+1wcd(hdc),', p_d,
                              p_d_next,
                              w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_next_d and valid_c and valid_c, 'pdwd+1pcd(hdc),', p_d,
                              w_d_next,
                              p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_next_d and valid_c and valid_c, 'pdwd+1wcd(hdc),', p_d,
                              w_d_next,
                              w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_next_d and valid_c and valid_c, 'wdpd+1pcd(hdc),', w_d,
                              p_d_next,
                              p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_next_d and valid_c and valid_c, 'wdpd+1wcd(hdc),', w_d,
                              p_d_next,
                              w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_next_d and valid_c and valid_c, 'wdwd+1pcd(hdc),', w_d,
                              w_d_next,
                              p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_next_d and valid_c and valid_c, 'wdwd+1wcd(hdc),', w_d,
                              w_d_next,
                              w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_d and valid_c and valid_c, 'pd-1pdpcd(hdc),', p_d_prev,
                              p_d,
                              p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_d and valid_c and valid_c, 'pd-1pdwcd(hdc),', p_d_prev,
                              p_d,
                              w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_d and valid_c and valid_c, 'pd-1wdpcd(hdc),', p_d_prev,
                              w_d,
                              p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_d and valid_c and valid_c, 'pd-1wdwcd(hdc),', p_d_prev,
                              w_d,
                              w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_d and valid_c and valid_c, 'wd-1pdpcd(hdc),', w_d_prev,
                              p_d,
                              p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_d and valid_c and valid_c, 'wd-1pdwcd(hdc),', w_d_prev,
                              p_d,
                              w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_d and valid_c and valid_c, 'wd-1wdpcd(hdc),', w_d_prev,
                              w_d,
                              p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_d and valid_c and valid_c, 'wd-1wdwcd(hdc),', w_d_prev,
                              w_d,
                              w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_c and valid_c, 'pdpc-1pcd(hdc),', p_d,
                              p_c_prev,
                              p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_c and valid_c, 'pdpc-1wcd(hdc),', p_d,
                              p_c_prev,
                              w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_c and valid_c, 'pdwc-1pcd(hdc),', p_d,
                              w_c_prev,
                              p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_c and valid_c, 'pdwc-1wcd(hdc),', p_d,
                              w_c_prev,
                              w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_c and valid_c, 'wdpc-1pcd(hdc),', w_d,
                              p_c_prev,
                              p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_c and valid_c, 'wdpc-1wcd(hdc),', w_d,
                              p_c_prev,
                              w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_c and valid_c, 'wdwc-1pcd(hdc),', w_d,
                              w_c_prev,
                              p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_c and valid_c and valid_c, 'wdwc-1wcd(hdc),', w_d,
                              w_c_prev,
                              w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_c and valid_next_c and valid_c, 'pdpcpc+1d(hdc),', p_d, p_c,
                              p_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_c and valid_next_c and valid_c, 'pdpcwc+1d(hdc),', p_d, p_c,
                              w_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_c and valid_next_c and valid_c, 'pdwcpc+1d(hdc),', p_d, w_c,
                              p_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_c and valid_next_c and valid_c, 'pdwcwc+1d(hdc),', p_d, w_c,
                              w_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_c and valid_next_c and valid_c, 'wdpcpc+1d(hdc),', w_d, p_c,
                              p_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_c and valid_next_c and valid_c, 'wdpcwc+1d(hdc),', w_d, p_c,
                              w_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_c and valid_next_c and valid_c, 'wdwcpc+1d(hdc),', w_d, w_c,
                              p_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_c and valid_next_c and valid_c, 'wdwcwc+1d(hdc),', w_d, w_c,
                              w_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_next_d and valid_prev_c and valid_c and valid_c,
                              'pdpd+1pc-1pcd(hdc),', p_d, p_d_next, p_c_prev, p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_next_d and valid_prev_c and valid_c and valid_c,
                              'pdpd+1pc-1wcd(hdc),', p_d, p_d_next, p_c_prev, w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_next_d and valid_prev_c and valid_c and valid_c,
                              'pdpd+1wc-1pcd(hdc),', p_d, p_d_next, w_c_prev, p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_next_d and valid_prev_c and valid_c and valid_c,
                              'pdpd+1wc-1wcd(hdc),', p_d, p_d_next, w_c_prev, w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_next_d and valid_prev_c and valid_c and valid_c,
                              'pdwd+1pc-1pcd(hdc),', p_d, w_d_next, p_c_prev, p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_next_d and valid_prev_c and valid_c and valid_c,
                              'pdwd+1pc-1wcd(hdc),', p_d, w_d_next, p_c_prev, w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_next_d and valid_prev_c and valid_c and valid_c,
                              'pdwd+1wc-1pcd(hdc),', p_d, w_d_next, w_c_prev, p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_next_d and valid_prev_c and valid_c and valid_c,
                              'pdwd+1wc-1wcd(hdc),', p_d, w_d_next, w_c_prev, w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_next_d and valid_prev_c and valid_c and valid_c,
                              'wdpd+1pc-1pcd(hdc),', w_d, p_d_next, p_c_prev, p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_next_d and valid_prev_c and valid_c and valid_c,
                              'wdpd+1pc-1wcd(hdc),', w_d, p_d_next, p_c_prev, w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_next_d and valid_prev_c and valid_c and valid_c,
                              'wdpd+1wc-1pcd(hdc),', w_d, p_d_next, w_c_prev, p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_next_d and valid_prev_c and valid_c and valid_c,
                              'wdpd+1wc-1wcd(hdc),', w_d, p_d_next, w_c_prev, w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_next_d and valid_prev_c and valid_c and valid_c,
                              'wdwd+1pc-1pcd(hdc),', w_d, w_d_next, p_c_prev, p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_next_d and valid_prev_c and valid_c and valid_c,
                              'wdwd+1pc-1wcd(hdc),', w_d, w_d_next, p_c_prev, w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_next_d and valid_prev_c and valid_c and valid_c,
                              'wdwd+1wc-1pcd(hdc),', w_d, w_d_next, w_c_prev, p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_next_d and valid_prev_c and valid_c and valid_c,
                              'wdwd+1wc-1wcd(hdc),', w_d, w_d_next, w_c_prev, w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_next_d and valid_c and valid_next_c and valid_c,
                              'pdpd+1pcpc+1d(hdc),', p_d, p_d_next, p_c, p_c_next, d_h_d + d_d_c)  # startt
                self._add_key(keys, valid_next_d and valid_c and valid_next_c and valid_c,
                              'pdpd+1pcwc+1d(hdc),', p_d, p_d_next, p_c, w_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_next_d and valid_c and valid_next_c and valid_c,
                              'pdpd+1wcpc+1d(hdc),', p_d, p_d_next, w_c, p_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_next_d and valid_c and valid_next_c and valid_c,
                              'pdpd+1wcwc+1d(hdc),', p_d, p_d_next, w_c, w_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_next_d and valid_c and valid_next_c and valid_c,
                              'pdwd+1pcpc+1d(hdc),', p_d, w_d_next, p_c, p_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_next_d and valid_c and valid_next_c and valid_c,
                              'pdwd+1pcwc+1d(hdc),', p_d, w_d_next, p_c, w_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_next_d and valid_c and valid_next_c and valid_c,
                              'pdwd+1wcpc+1d(hdc),', p_d, w_d_next, w_c, p_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_next_d and valid_c and valid_next_c and valid_c,
                              'pdwd+1wcwc+1d(hdc),', p_d, w_d_next, w_c, w_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_next_d and valid_c and valid_next_c and valid_c,
                              'wdpd+1pcpc+1d(hdc),', w_d, p_d_next, p_c, p_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_next_d and valid_c and valid_next_c and valid_c,
                              'wdpd+1pcwc+1d(hdc),', w_d, p_d_next, p_c, w_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_next_d and valid_c and valid_next_c and valid_c,
                              'wdpd+1wcpc+1d(hdc),', w_d, p_d_next, w_c, p_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_next_d and valid_c and valid_next_c and valid_c,
                              'wdpd+1wcwc+1d(hdc),', w_d, p_d_next, w_c, w_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_next_d and valid_c and valid_next_c and valid_c,
                              'wdwd+1pcpc+1d(hdc),', w_d, w_d_next, p_c, p_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_next_d and valid_c and valid_next_c and valid_c,
                              'wdwd+1pcwc+1d(hdc),', w_d, w_d_next, p_c, w_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_next_d and valid_c and valid_next_c and valid_c,
                              'wdwd+1wcpc+1d(hdc),', w_d, w_d_next, w_c, p_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_next_d and valid_c and valid_next_c and valid_c,
                              'wdwd+1wcwc+1d(hdc),', w_d, w_d_next, w_c, w_c_next, d_h_d + d_d_c)  # finish
                self._add_key(keys, valid_prev_d and valid_prev_c and valid_c and valid_c,
                              'pd-1pdpc-1pcd(hdc),', p_d_prev, p_d, p_c_prev, p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_d and valid_prev_c and valid_c and valid_c,
                              'pd-1pdpc-1wcd(hdc),', p_d_prev, p_d, p_c_prev, w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_d and valid_prev_c and valid_c and valid_c,
                              'pd-1pdwc-1pcd(hdc),', p_d_prev, p_d, w_c_prev, p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_d and valid_prev_c and valid_c and valid_c,
                              'pd-1pdwc-1wcd(hdc),', p_d_prev, p_d, w_c_prev, w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_d and valid_prev_c and valid_c and valid_c,
                              'pd-1wdpc-1pcd(hdc),', p_d_prev, w_d, p_c_prev, p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_d and valid_prev_c and valid_c and valid_c,
                              'pd-1wdpc-1wcd(hdc),', p_d_prev, w_d, p_c_prev, w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_d and valid_prev_c and valid_c and valid_c,
                              'pd-1wdwc-1pcd(hdc),', p_d_prev, w_d, w_c_prev, p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_d and valid_prev_c and valid_c and valid_c,
                              'pd-1wdwc-1wcd(hdc),', p_d_prev, w_d, w_c_prev, w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_d and valid_prev_c and valid_c and valid_c,
                              'wd-1pdpc-1pcd(hdc),', w_d_prev, p_d, p_c_prev, p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_d and valid_prev_c and valid_c and valid_c,
                              'wd-1pdpc-1wcd(hdc),', w_d_prev, p_d, p_c_prev, w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_d and valid_prev_c and valid_c and valid_c,
                              'wd-1pdwc-1pcd(hdc),', w_d_prev, p_d, w_c_prev, p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_d and valid_prev_c and valid_c and valid_c,
                              'wd-1pdwc-1wcd(hdc),', w_d_prev, p_d, w_c_prev, w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_d and valid_prev_c and valid_c and valid_c,
                              'wd-1wdpc-1pcd(hdc),', w_d_prev, w_d, p_c_prev, p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_d and valid_prev_c and valid_c and valid_c,
                              'wd-1wdpc-1wcd(hdc),', w_d_prev, w_d, p_c_prev, w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_d and valid_prev_c and valid_c and valid_c,
                              'wd-1wdwc-1pcd(hdc),', w_d_prev, w_d, w_c_prev, p_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_d and valid_prev_c and valid_c and valid_c,
                              'wd-1wdwc-1wcd(hdc),', w_d_prev, w_d, w_c_prev, w_c, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_d and valid_c and valid_next_c and valid_c,
                              'pd-1pdpcpc+1d(hdc),', p_d_prev, p_d, p_c, p_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_d and valid_c and valid_next_c and valid_c,
                              'pd-1pdpcwc+1d(hdc),', p_d_prev, p_d, p_c, w_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_d and valid_c and valid_next_c and valid_c,
                              'pd-1pdwcpc+1d(hdc),', p_d_prev, p_d, w_c, p_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_d and valid_c and valid_next_c and valid_c,
                              'pd-1pdwcwc+1d(hdc),', p_d_prev, p_d, w_c, w_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_d and valid_c and valid_next_c and valid_c,
                              'pd-1wdpcpc+1d(hdc),', p_d_prev, w_d, p_c, p_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_d and valid_c and valid_next_c and valid_c,
                              'pd-1wdpcwc+1d(hdc),', p_d_prev, w_d, p_c, w_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_d and valid_c and valid_next_c and valid_c,
                              'pd-1wdwcpc+1d(hdc),', p_d_prev, w_d, w_c, p_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_d and valid_c and valid_next_c and valid_c,
                              'pd-1wdwcwc+1d(hdc),', p_d_prev, w_d, w_c, w_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_d and valid_c and valid_next_c and valid_c,
                              'wd-1pdpcpc+1d(hdc),', w_d_prev, p_d, p_c, p_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_d and valid_c and valid_next_c and valid_c,
                              'wd-1pdpcwc+1d(hdc),', w_d_prev, p_d, p_c, w_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_d and valid_c and valid_next_c and valid_c,
                              'wd-1pdwcpc+1d(hdc),', w_d_prev, p_d, w_c, p_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_d and valid_c and valid_next_c and valid_c,
                              'wd-1pdwcwc+1d(hdc),', w_d_prev, p_d, w_c, w_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_d and valid_c and valid_next_c and valid_c,
                              'wd-1wdpcpc+1d(hdc),', w_d_prev, w_d, p_c, p_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_d and valid_c and valid_next_c and valid_c,
                              'wd-1wdpcwc+1d(hdc),', w_d_prev, w_d, p_c, w_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_d and valid_c and valid_next_c and valid_c,
                              'wd-1wdwcpc+1d(hdc),', w_d_prev, w_d, w_c, p_c_next, d_h_d + d_d_c)
                self._add_key(keys, valid_prev_d and valid_c and valid_next_c and valid_c,
                              'wd-1wdwcwc+1d(hdc),', w_d_prev, w_d, w_c, w_c_next, d_h_d + d_d_c)

        # extended feature list

        # distance between
        self._add_key(keys, True, f'dist{trg_ind-src_ind}_src_tag_trg', src_tag, trg_tag, d_h_d)
        #
        # suffix features
        suffix_checker = lambda src_word, suffix: len(src_word) > len(suffix) and src_word[-len(suffix):] == suffix
        # base suffix
        [self._add_key(keys, True, f'suffix_base_src', src_word) for suffix in suffix_list_base if
         suffix_checker(src_word, suffix)]
        [self._add_key(keys, True, f'suffix_base_trg', trg_word) for suffix in suffix_list_base if
         suffix_checker(trg_word, suffix)]
        [self._add_key(keys, True, f'suffix_base_src_word_trg', src_word, trg_word) for suffix in suffix_list_base if
         suffix_checker(src_word, suffix)]
        [self._add_key(keys, True, f'suffix_base_src_tag_trg', src_word, trg_tag) for suffix in suffix_list_base if
         suffix_checker(trg_word, suffix)]
        # verb suffix
        [self._add_key(keys, True, f'suffix_verb_src', src_word) for suffix in suffix_list_verbs if
         suffix_checker(src_word, suffix)]
        [self._add_key(keys, True, f'suffix_verb_trg', trg_word) for suffix in suffix_list_verbs if
         suffix_checker(trg_word, suffix)]
        [self._add_key(keys, True, f'suffix_verb_src_word_trg', src_word, trg_word) for suffix in suffix_list_verbs if
         suffix_checker(src_word, suffix)]
        [self._add_key(keys, True, f'suffix_verb_src_tag_trg', src_word, trg_tag) for suffix in suffix_list_verbs if
         suffix_checker(trg_word, suffix)]
        # adj suffix
        [self._add_key(keys, True, f'suffix_adj_src', src_word) for suffix in suffix_list_adj if
         suffix_checker(src_word, suffix)]
        [self._add_key(keys, True, f'suffix_adj_trg', trg_word) for suffix in suffix_list_adj if
         suffix_checker(trg_word, suffix)]
        [self._add_key(keys, True, f'suffix_adj_src_word_trg', src_word, trg_word) for suffix in suffix_list_adj if
         suffix_checker(src_word, suffix)]
        [self._add_key(keys, True, f'suffix_adj_src_tag_trg', src_word, trg_tag) for suffix in suffix_list_adj if
         suffix_checker(trg_word, suffix)]
        # adverb suffix
        [self._add_key(keys, True, f'suffix_adverb_src', src_word) for suffix in suffix_list_adverbs if
         suffix_checker(src_word, suffix)]
        [self._add_key(keys, True, f'suffix_adverb_trg', trg_word) for suffix in suffix_list_adverbs if
         suffix_checker(trg_word, suffix)]
        [self._add_key(keys, True, f'suffix_adverb_src_word_trg', src_word, trg_word) for suffix in suffix_list_adverbs
         if suffix_checker(src_word, suffix)]
        [self._add_key(keys, True, f'suffix_adverb_src_tag_trg', src_word, trg_tag) for suffix in suffix_list_adverbs if
         suffix_checker(trg_word, suffix)]

        # prefix templates
        prefix_checker = lambda src_word, prefix: len(src_word) > len(prefix) and src_word[
                                                                                  :len(prefix)].lower() == prefix
        [self._add_key(keys, True, f'prefix_tag_src', prefix, src_word) for prefix in prefix_list if
         prefix_checker(src_word, prefix)]
        [self._add_key(keys, True, f'prefix_tag_trg', prefix, trg_word) for prefix in prefix_list if
         prefix_checker(trg_word, prefix)]
        [self._add_key(keys, True, f'prefix_src_word_trg', prefix, src_word, trg_word) for prefix in prefix_list
         if prefix_checker(src_word, prefix)]
        [self._add_key(keys, True, f'prefix_src_tag_trg', prefix, src_word, trg_tag) for prefix in prefix_list if
         prefix_checker(trg_word, prefix)]

        # capital letters
        # first letter
        self._add_key(keys, (src_word[0].isupper() and src_ind != 1), f'upper_tag_src', src_tag)
        self._add_key(keys, src_word[0].isupper() and src_ind != 1, f'upper_tag_src_tag_trg', src_tag, trg_tag)

        # contains digit
        self._add_key(keys, any(c.isdigit() for c in src_word), f'contains_digit_tag_src_tag_trg', src_tag, trg_tag)
        self._add_key(keys, any(c.isdigit() for c in trg_word), f'tag_src_contains_digit_tag_trg', src_tag, trg_tag)

        return keys

    def create_init_graph(self, obj):
        full_graph = {}
        sentence_len = len(obj.sentence)
        # debug_count = 0  # TODO: remove after debug
        for src in range(sentence_len):
            full_graph[src] = []
            for trg in range(sentence_len):
                if self.features_full[self.get_key(f'tag_src_tag_trg', obj.tags[src], obj.tags[trg])]:
                    full_graph[src].append(trg)  # TODO: save in dictionary
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

    def get_child_list(self, h_ind, d_ind, graph):
        h_childs = []
        if h_ind in graph:
            h_childs = graph[h_ind]
        return [child for child in h_childs + graph[d_ind] if child != h_ind]
        # if h_ind in graph and graph[h_ind] != []:
        #     return graph[h_ind][0]
        # elif len(graph[d_ind]) > 1:
        #     for sibling in graph[d_ind]:
        #         if sibling != h_ind:
        #             return sibling
        # else:
        #     return []


class BootCamp:

    def __init__(self, features):

        assert isinstance(features, Features)
        self.features = features

    def investigate_soldiers(self, soldier_list: list, verbose: bool = True):
        """

        :param verbose: Display Progress bar
        :type verbose: bool
        :param soldier_list:
        :type soldier_list:
        """
        print("Investigating Soldiers")
        for soldier in tqdm(soldier_list, leave=False, disable=not verbose, unit=' Soldier', mininterval=5):
            self.features.extract_features(soldier)

    def truncate_features(self, n_top, n_bottom=None):
        self.features.truncate_features(n_top, n_bottom)

    def train_soldiers(self, soldier_list, fast=True, verbose: bool = True):
        """
        Create feature tensor for each object
        :param soldier_list:
        :type soldier_list:
        :return:
        :rtype:
        """
        # if no truncation has been made, generate key2token
        print('Training soldiers')
        if len(list(self.features.key2token.keys())) == 0:
            self.features.tokenize()
        # fill tensor
        for soldier in tqdm(soldier_list, leave=False, disable=not verbose, unit=' Soldier', mininterval=5):
            self.features.fill_tensor(soldier, fast=fast)

        # return soldier_list  # inplace

    def get_model(self):
        return self.features.model_type
