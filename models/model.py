import pandas as pd
import numpy as np
from models.features import FinkMos
from scipy.optimize import minimize

class Model:
    def __init__(self, tests):
        self.tests = tests
        self.v = np.ones(len(tests))  # TODO: init wisely ?
        self.x = None
        self.y = None
        self.vector_x_y = None
        self.tag_corpus = None
        self.tag_corpus_tokenized = None
        self.strin2token =dict()
        self.token2string = dict()

    def fit(self, x, y, learning_rate=0.02, x_val=None, y_val=None):
        """
        Fit will train the Model
            - Encoding
                - For the data will create a Tensor [ # todo Read more
            - Gradient decent
                -Loss
                - update V
            - Calculate metrics

        :param x: DataFrame [row = sentence , col = word]
        :param y: DataFrame [row = sentence tag , col = Word tag]
        :param learning_rate: [don't know if need it] # TODO check if remove
        :param x_val:[row = sentence , col = word]
        :param y_val:[row = sentence tag , col = Word tag]
        :return: metrics dict {} $# TODO check witch metrics we need
        """
        assert isinstance(x, pd.DataFrame)
        assert isinstance(y, pd.DataFrame)
        self.x = x
        self.y = y
        self.tag_corpus = pd.unique(y.values.ravel('K'))  # TODO remove '*' , '<PAD>' , '<STOP>"
        self._vectorize()

        return

    def predict(self, x):
        """
        This will work with the Viterbi
        :param x: [sentences * words]
        :return: matrix [ sentence_tags * words]
        """
        # validity check
        tokenized_ans = self._viterbi(x, self.tag_corpus_tokenized)
        # translate to tags
        tag_ans = tokenized_ans  # TODO
        return tag_ans

        pass

    def model_function(self, next_tag, word_num, previous_tags, sentence):
        """
        :param next_tag: Next tag
        :type next_tag: int
        :param word_num: Number of word in the sentence
        :type word_num: int
        :param previous_tags: [t_-2, t_-1] first index list of -2 position tag, second tag for -1 position tag
        :type previous_tags: [int, int]
        :param sentence: List of words
        :type sentence: np.array ['*','*', 'first','second', ..... ,'<STOP>', '<PAD>']
        :return: List of Probabilities of next_tag
        :rtype: List of Floats
        """


    def _viterbi(self, sentence, all_tags):
        """
        :param sentence: List of words
        :type sentence: List
        :param all_tags: List of possible tags (ordered in the same order used for the learning) #TODO: consider passing num of tags
        :type all_tags: List
        :return: List of tags
        :rtype: List
        """
        num_words = len(sentence)
        num_tags = len(all_tags)
        dims = (num_words, num_tags, num_tags)
        p_table = np.empty(dims, dtype=np.float)  # pi(k,u,v) - maximum probability of tag sequence


        # ending in tags u,v at position k
        p_table[0, 0, 0] = 1  # init
        bp_table = np.empty(dims, dtype=np.int8)
        answer = [None] * num_words
        for k in range(num_words):
            if k == 1:
                tags1 = [0]
            elif k < len(sentence) - 2:
                tags1 = all_tags
            for t1 in range(len(tags1)):
                for t2 in range(len(all_tags)):
                    if k == 1 or k == 2:  # 0 -> '*' tag
                        w = [0]
                    else:
                        w = range(len(all_tags))
                    options = p_table[k - 1, w, t1] * self.model_function(next_tag=t2, word_num=k,
                                                                          previous_tags=[w, t1], sentence=sentence)
                    bp_table[k, t1, t2] = np.argmax(options)
                    p_table[k, t1, t2] = options[bp_table[k, t1, t2]]
        answer[num_words - 2], answer[num_words - 1] = np.unravel_index(bp_table[num_words - 1, :, :].argmax(),
                                                                        bp_table[num_words - 1, :, :].shape)

        for k in reversed(range(num_words - 2)):
            answer[k] = bp_table[k + 2, answer[k + 1], answer[k + 2]]

        return answer

    def _vectorize(self):

        vectors = []
        matrix = []
        for i in range(self.x.shape[0]):
            a = FinkMos(self.x.loc[i, :], self.y.loc[i, :], tests=self.tests, tag_corpus=self.tag_corpus)
            vectors.append(a.fill_test())
            matrix.append(a.f_x_y)
        self.vector_x_y = np.array(vectors, dtype=FinkMos)

        self.lin_loss_matrix_x_y = np.concatenate(matrix, axis=0)
        # is a sentence

    def _loss(self, v):
        positive = self._calculate_positive(v)
        non_linear = self._calculate_nonlinear(v)
        penalty = 0.5 * np.linalg.norm(v)

        return positive - non_linear + penalty

    def _calculate_positive(self, v):
        """
        This method will solve the positive part of the loss Function
        for all the sentence
        sum (sum (v * f(h_i^(k),y_i), for i=0 to max sise word), for k =0 to last sentence)
        = sum( F dot v ) where F is concatenate matrix for all the vectors f
        :param v:
        :return:
        """
        assert isinstance(v, np.ndarray)

        dot_m = self.lin_loss_matrix_x_y @ v
        return dot_m.sum()

    def _calculate_nonlinear(self, v):
        assert isinstance(v, np.ndarray)
        matrix = []
        for mat in self.vector_x_y:
            assert isinstance(mat, FinkMos)
            matrix.append(mat.sentence_non_lineard_loss(v))
        return np.sum(matrix)
