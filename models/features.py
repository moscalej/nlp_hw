from models.prerocesing import PreprocessTags
import yaml
import pickle
import dill as pickle
import pandas as pd


# IN_DT_NN        1825
# NN_<STOP>_*     1752
# DT_JJ_NN        1590
# DT_NN_IN        1423
# NN_IN_DT        1349
# NNP_NNP_NNP     1192
# *_*_DT          1134
# *_*_NNP         1083
# IN_DT_JJ        1018
# NNS_<STOP>_*     990
# JJ_NN_IN         886
# NNP_NNP_,        879
# NN_IN_NNP        810
# IN_NNP_NNP       806
# *_*_IN           665
# *_NNP_NNP        663
# NNS_IN_DT        663

class Features:
    def __init__(self):
        # self.corpus_size = len(x)
        # self.x = pd.Series(x)
        # self.y = pd.Series(y)
        # self.y_corpus = y.unique()
        self.lambdas = dict()
        self.tuple_corpus = None

        # if files exists load:
        # lambdas
        # full_list
        #

    def generate_tuple_corpus(self, x, y):
        if x is None:
            return None
        tup_list = []
        for ind in range(len(x)):
            if ind in [0, 1]:
                continue
            tup = (y[ind], y[ind - 1], y[ind - 2], x[ind], x[ind - 1], x[ind - 2])
            tup_list.append(tup)
            self.tuple_corpus = tup_list

    def add_lambdas(self, lambdas_dict):
        self.lambdas.update(lambdas_dict)

    def get_tests(self, path=fr"../training/report_lambdas_dict.p"):
        result = self.lambdas
        if len(result) == 0:
            with open(path, 'rb') as stream:
                self.add_lambdas(pickle.load(stream))
        return result

    def generate_lambdas(self, template, tuple_corpus=None):
        if tuple_corpus is None:
            tuple_corpus = self.tuple_corpus
        template_name = template.__name__
        for tup in tuple_corpus:
            name, func = template(tup)
            if name in self.lambdas:
                self.lambdas[name]['tup_list'].append(tup)
            else:
                lambda_dict = {name: {'func': template(tup), 'tup_list': [tup]}}
                self.add_lambdas(lambda_dict)


trigrams = dict(

    tri_000=lambda sentence, place, y, y_1, y_2:
    1 if y_2 == 'IN' and y_1 == 'DT' and y == 'NN' else 0,
    tri_001=lambda sentence, place, y, y_1, y_2:
    1 if y_2 == 'DT' and y_1 == 'JJ' and y == 'NN' else 0,
    tri_002=lambda sentence, place, y, y_1, y_2:
    1 if y_2 == 'DT' and y_1 == 'NN' and y == 'IN' else 0,
    tri_003=lambda sentence, place, y, y_1, y_2:
    1 if y_2 == 'NN' and y_1 == 'IN' and y == 'DT' else 0,
    tri_004=lambda sentence, place, y, y_1, y_2:
    1 if y_2 == 'NNP' and y_1 == 'NNP' and y == 'MNP' else 0,
    tri_005=lambda sentence, place, y, y_1, y_2:
    1 if y_2 == 'JJ' and y_1 == 'NN' and y == 'IN' else 0,
    tri_006=lambda sentence, place, y, y_1, y_2:
    1 if y_2 == 'NN' and y_1 == 'IN' and y == 'NP' else 0,

)

# <STOP> *      5000
# DT NN         4934
# NNP NNP       4636
# NN IN         4224
# IN DT         4162
# JJ NN         3498
# DT JJ         2370
# NN NN         2158
# IN NNP        2097
# NNS IN        1912
# NN ,          1847
# JJ NNS        1841
# NN <STOP>     1752
# TO VB         1701
# NNP ,         1642
# IN NN         1352
# NN NNS        1291

biagrams = dict(
    bi_000=lambda sentence, place, y, y_1, y_2:
    1 if y_1 == 'DT' and y == 'NN' else 0,
    bi_001=lambda sentence, place, y, y_1, y_2:
    1 if y_1 == 'NNP' and y == 'NNP' else 0,
    bi_002=lambda sentence, place, y, y_1, y_2:
    1 if y_1 == 'NN' and y == 'IN' else 0,
    bi_003=lambda sentence, place, y, y_1, y_2:
    1 if y_1 == 'IN' and y == 'DT' else 0,
    bi_004=lambda sentence, place, y, y_1, y_2:
    1 if y_1 == 'NNP' and y == 'NNP' else 0,
    bi_005=lambda sentence, place, y, y_1, y_2:
    1 if y_1 == 'NN' and y == 'IN' else 0,
    bi_006=lambda sentence, place, y, y_1, y_2:
    1 if y_1 == 'JJ' and y == 'NN' else 0,
)

own_func = dict(

    own_000=lambda sentence, place, y, y_1, y_2:
    1 if sentence[place] == 'the' and y == 'IN' else 0,

    own_001=lambda sentence, place, y, y_1, y_2:
    1 if sentence[place] == 'about' and y == 'IN' else 0,

    own_002=lambda sentence, place, y, y_1, y_2:
    1 if y_1 == 'IN' and y == 'JJ' else 0,

    own_003=lambda sentence, place, y, y_1, y_2:
    1 if y_1 == 'IN' and y_2 == 'NNS' and y == 'JJ' else 0,

    own_004=lambda sentence, place, y, y_1, y_2:  # rework
    1 if sentence[place].lower() == 'the' and y == 'DT' else 0,

    own_005=lambda sentence, place, y, y_1, y_2:
    1 if sentence[place] == 'the' and y == 'IN' else 0,

    own_006=lambda sentence, place, y, y_1, y_2:
    1 if sentence[place] == 'is' and y == 'VBS' else 0,

    own_007=lambda sentence, place, y, y_1, y_2:
    1 if sentence[place] == 'of' and y == 'IN' else 0,

    own_009=lambda sentence, place, y, y_1, y_2:
    1 if sentence[place] == 'to' and y == 'TO' else 0,

    own_010=lambda sentence, place, y, y_1, y_2:
    1 if sentence[place] == 'a' and y == 'DT' else 0,

    own_011=lambda sentence, place, y, y_1, y_2:
    1 if sentence[place] == 'and' and y == 'CC' else 0,

    own_012=lambda sentence, place, y, y_1, y_2:
    1 if sentence[place] == "'s" and y == 'POS' else 0,

    own_013=lambda sentence, place, y, y_1, y_2:
    1 if sentence[place] == 'is' and y == 'VBZ' else 0,

    # capital letter features:
    own_014=lambda sentence, place, y, y_1, y_2:  # first letter is capital letter
    1 if sentence[place][0] != sentence[place][0].lower else 0,

    own_015=lambda sentence, place, y, y_1, y_2:  # capital letter not in the beginning
    1 if sentence[place][1:-1] != sentence[place][1:-1].lower else 0,

)

rare_func = dict(
    # Prefix (un, re, in, im, dis, dif, en, em, pre, mis, a)
    rar_000=lambda sentence, place, y, y_1, y_2:
    1 if sentence[place][0:1].lower() == 'un' else 0,

    rar_001=lambda sentence, place, y, y_1, y_2:
    1 if sentence[place][0:1].lower() == 're' else 0,

    rar_002=lambda sentence, place, y, y_1, y_2:
    1 if sentence[place][0:1].lower() == 'in' else 0,

    rar_003=lambda sentence, place, y, y_1, y_2:
    1 if sentence[place][0:1].lower() == 'im' else 0,

    rar_004=lambda sentence, place, y, y_1, y_2:
    1 if sentence[place][0:2].lower() == 'dis' else 0,

    rar_005=lambda sentence, place, y, y_1, y_2:
    1 if sentence[place][0:2].lower() == 'dif' else 0,

    rar_006=lambda sentence, place, y, y_1, y_2:
    1 if sentence[place][0:1].lower() == 'en' else 0,

    rar_007=lambda sentence, place, y, y_1, y_2:
    1 if sentence[place][0:1].lower() == 'em' else 0,

    rar_008=lambda sentence, place, y, y_1, y_2:
    1 if sentence[place][0:2].lower() == 'pre' else 0,

    rar_009=lambda sentence, place, y, y_1, y_2:
    1 if sentence[place][0:2].lower() == 'mis' else 0,

    rar_010=lambda sentence, place, y, y_1, y_2:
    1 if sentence[place][0:0].lower() == 'a' else 0,
)

# 'to'
# 'in' - (particle words)
# 'can't'

# Rare features

# contains a number
# contains hyphen '-'

# Verb Forms
# has\have\had been , ed

# Particle verbs
# in\on\up', 'o\ "sink", "hang"


rapnapak = dict(
    f_100=lambda sentence, place, y, y_1, y_2: \
        1 if sentence[place] == 'base' and y == 'Vt' else 0,

    f_101=lambda sentence, place, y, y_1, y_2: \
        1 if len(sentence[place]) > 4 and sentence[place][-3:] == 'ing' and y == 'VBG' else 0,

    f_102=lambda sentence, place, y, y_1, y_2: \
        1 if len(sentence[place]) > 4 and sentence[place][:3] == 'pre' and y == 'NN' else 0,

    f_103=lambda sentence, place, y, y_1, y_2: 1 if y == 'Vt' and y_1 == 'JJ' and y_2 == 'DT' else 0,

    f_104=lambda sentence, place, y, y_1, y_2: 1 if y == 'Vt' and y_1 == 'JJ' else 0,

    f_105=lambda sentence, place, y, y_1, y_2: 1 if y == 'Vt' else 0,

    f_106=lambda sentence, place, y, y_1, y_2: 1 if sentence[place - 1] == 'the' and y == 'Vt' else 0,

    f_107=lambda sentence, place, y, y_1, y_2: \
        1 if place < sentence.size - 1 and sentence[place + 1] == 'the' and y == 'Vt' else 0,
)


########################################################################################################
# Feature Templates
def template_suffix(suffix_length, suffix, tag):
    # tag0 : input[0] fov[0]
    # tag_1 : input[1] fov[1]
    # tag_2 : input[2] fov[2]
    # word_0 : input[3]  fov[3]
    # word_1 : input[4]  fov[4]
    # word_2 : input[5]  fov[5]
    res_func = lambda fov: \
        1 if len(fov[3]) > suffix_length and \
             fov[3][(-suffix_length):].lower() == suffix and \
             fov[0] == tag else 0
    return res_func


def template_prefix(prefix_length, prefix, tag):
    # tag0 : input[0] fov[0]
    # tag_1 : input[1] fov[1]
    # tag_2 : input[2] fov[2]
    # word_0 : input[3]  fov[3]
    # word_1 : input[4]  fov[4]
    # word_2 : input[5]  fov[5]
    name = f'w_t-{input[0]}_^_^_{input[3]}_^_^'
    res_func = lambda fov: \
        len(fov[3]) > prefix_length and \
        fov[3][0:prefix_length].lower() == prefix and \
        fov[0] == tag
    return res_func


def template_w_t(input):  # <w, t>
    # tag0 : input[0] fov[0]
    # tag_1 : input[1] fov[1]
    # tag_2 : input[2] fov[2]
    # word_0 : input[3]  fov[3]
    # word_1 : input[4]  fov[4]
    # word_2 : input[5]  fov[5]
    name = f'w_t-{input[0]}_^_^_{input[3]}_^_^'
    func = lambda fov: fov[3] == input[3] and fov[0] == input[0]
    return name, func


def test_feature_template(data_x, data_y):
    # get tuples
    # get test_list
    #
    pass


def template_w_w_1_t_t_1(input):  # <w, t>
    # tag0 : input[0] fov[0]
    # tag_1 : input[1] fov[1]
    # tag_2 : input[2] fov[2]
    # word_0 : input[3]  fov[3]
    # word_1 : input[4]  fov[4]
    # word_2 : input[5]  fov[5]
    name = f'w_w_1_t_t_1-{input[0]}_{input[1]}_^_{input[3]}_{input[4]}_^'
    func = lambda fov: fov[3] == input[3] and fov[0] == input[0] and fov[4] == input[4] \
                       and fov[1] == input[1]
    return name, func


def template_w_w_1_w_2_t_t_1_t_2(input):  # <w, t>
    # tag0 : input[0] fov[0]
    # tag_1 : input[1] fov[1]
    # tag_2 : input[2] fov[2]
    # word_0 : input[3]  fov[3]
    # word_1 : input[4]  fov[4]
    # word_2 : input[5]  fov[5]
    name = f'w_w_1_t_t_1-{input[0]}_{input[1]}_{input[2]}_{input[3]}_{input[4]}_{input[5]}'
    func = lambda fov: fov[3] == input[3] and fov[0] == input[0] and \
                       fov[4] == input[4] and fov[1] == input[1] and \
                       fov[5] == input[5] and fov[2] == input[2]
    return name, func


#  take 25 most frequent words, and 10 most frequent tags and iterate over all variations
# template_w_t = [(,), (,)]

# def template_t_1_t(tag_1, tag):  # <t_1, t>
#     res_func = lambda sentence, place, y, y_1, y_2: \
#                    1 if y_1 == tag_1 and y == tag else 0,
#     return res_func
#
#
# def template_t_2_t(tag_2, tag):  # <t_2, t>
#     res_func = lambda sentence, place, y, y_1, y_2: \
#                    1 if y_2 == tag_2 and y == tag else 0,
#     return res_func
#
#
# def template_w_2_w_1(word_2, word_1):  # <w_2, w_1>
#     res_func = lambda sentence, place, y, y_1, y_2: \
#                    1 if sentence[place - 2] == word_2 and sentence[place - 1] == word_1 else 0,
#     return res_func
#
#
# def template_w_3_w_2(word_3, word_2):  # <w_2, w_1>
#     res_func = lambda sentence, place, y, y_1, y_2: \
#                    1 if sentence[place - 3] == word_3 and sentence[place - 2] == word_2 else 0,
#     return res_func


w_2_w_1_list = [['have', 'been'], ['has', 'been'], ['had', 'been']]
w_3_w_2_list = [['have', 'been'], ['has', 'been'], ['had', 'been']]
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
frequent_tags = ["NN", "IN", "JJ", "DT", "NNS", "CC", "VBN", "RB", "VBD", "CD", "VBZ"]

# suffix_funcs_all = {}
# prefix_funcs_all = {}
# data = PreprocessTags(True).load_data(
#     r'..\data\train2.wtag')
# y_corpus = pd.Series(data.y).unique()
#
# for tag in y_corpus:
#     suffix_funcs = {f"suffix_{suffix}_{tag}": template_suffix(len(suffix), suffix, tag) for suffix in all_suffix}
#     suffix_funcs_all = {**suffix_funcs_all, **suffix_funcs}
#     prefix_funcs = {f"prefix_{prefix}_{tag}": template_prefix(len(prefix), prefix, tag) for prefix in prefix_list}
#     prefix_funcs_all = {**prefix_funcs_all, **prefix_funcs}

templates_dict = dict({})
# Format: {'template_w_t': {'template': template_w_t, 'tuples': None}}
dict_entry_gen = lambda name, func, tuples=None: {name: {'func': func, 'tuples': tuples}}
templates_dict.update(dict_entry_gen('template_w_t', template_w_t))  # DONE
templates_dict.update(dict_entry_gen('template_w_w_1_t_t_1', template_w_w_1_t_t_1))  # DONE
templates_dict.update(dict_entry_gen('template_w_w_1_w_2_t_t_1_t_2', template_w_w_1_w_2_t_t_1_t_2))  # DONE
