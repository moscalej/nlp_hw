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

    oun_000=lambda sentence, place, y, y_1, y_2:
    1 if sentence[place] == 'the' and y == 'IN' else 0,

    oun_001=lambda sentence, place, y, y_1, y_2:
    1 if sentence[place] == 'about' and y == 'IN' else 0,

    oun_002=lambda sentence, place, y, y_1, y_2:
    1 if y_1 == 'IN' and y == 'JJ' else 0,

    oun_003=lambda sentence, place, y, y_1, y_2:
    1 if y_1 == 'IN' and y_2 == 'NNS' and y == 'JJ' else 0,

    oun_004=lambda sentence, place, y, y_1, y_2:
    1 if sentence[place] == 'the' and y == 'IN' else 0,

    oun_005=lambda sentence, place, y, y_1, y_2:
    1 if sentence[place] == 'the' and y == 'IN' else 0,
)


class Features:
    def get_tests(self):
        functions = dict(
            f_100=self.f_100,
            f_101=self.f_101,
            f_102=self.f_102,
            f_103=self.f_103,
            f_104=self.f_104,
            f_105=self.f_105,
            f_106=self.f_106,
            f_107=self.f_107,
        )
        functions.update(trigrams)
        functions.update(biagrams)
        functions.update(own_func)
        return functions

    def f_100(self, sentence, place, y, y_1, y_2):
        return 1 if sentence[place] == 'base' and y == 'Vt' else 0

    def f_101(self, sentence, place, y, y_1, y_2):
        if len(sentence[place]) < 4:
            return 0
        return 1 if sentence[place][-3:] == 'ing' and y == 'VBG' else 0

    def f_102(self, sentence, place, y, y_1, y_2):
        if len(sentence[place]) < 4:
            return 0
        return 1 if sentence[place][:3] == 'pre' and y == 'NN' else 0

    def f_103(self, sentence, place, y, y_1, y_2):
        return 1 if y == 'Vt' and y_1 == 'JJ' and y_2 == 'DT' else 0

    def f_104(self, sentence, place, y, y_1, y_2):
        return 1 if y == 'Vt' and y_1 == 'JJ' else 0

    def f_105(self, sentence, place, y, y_1, y_2):
        return 1 if y == 'Vt' else 0

    def f_106(self, sentence, place, y, y_1, y_2):
        return 1 if sentence[place - 1] == 'the' and y == 'Vt' else 0

    def f_107(self, sentence, place, y, y_1, y_2):
        if place == sentence.size - 1:
            return 0
        return 1 if sentence[place + 1] == 'the' and y == 'Vt' else 0
