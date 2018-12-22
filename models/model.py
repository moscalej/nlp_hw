import numpy as np
import pandas as pd
from scipy.optimize import minimize

from models.score import Score
from models.sentence_processor import FinkMos


class Model:
    def __init__(self, tests):
        self.tests = tests
        self.v = np.random.uniform(-0.5, 0.5, len(tests))  # TODO: init wisely ?
        self.x = None
        self.y = None
        self.vector_x_y = None
        self.tag_corpus = None
        self.tag_corpus_tokenized = None
        self.strin2token = dict()
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
        assert isinstance(x, pd.Series)
        assert isinstance(y, pd.Series)
        self.x = x
        self.y = y
        base_corpus = pd.Series(['*', '<STOP>'])
        tag_corpus = pd.Series(y.value_counts().drop(['*', '<STOP>']).index)
        self.tag_corpus = base_corpus.append(tag_corpus)
        self.tag_corpus_tokenized = range(len(self.tag_corpus))
        self._translation()  # create dictionaries for tokenizing
        self._vectorize()
        # TODO: consider adding a test removal mechanism (from self.tests)
        self.opt_result = minimize(self._loss, np.ones(len(self.tests)), options=dict(disp=True), method='BFGS')
        self.v = self.opt_result['x']

        return

    def predict(self, x):
        """
        This will work with the Viterbi
        :param x: [sentences * words]
        :return: matrix [ sentence_tags * words]
        """
        # TODO: decide about sentence format (['*','*'..])
        # validity check
        # check x type
        # check sentence format
        #
        tokenized_ans = self._viterbi(x)

        # translate to tags
        tag_ans = [self.token2string[token] for token in tokenized_ans]
        # assert isinstance(tag_ans, pd.DataFrame)
        return tag_ans

    def confusion(self, x, y):
        """

        :param x:
        :type x:
        :param y:
        :type y:
        :return:
        :rtype:
        """
        assert isinstance(x, pd.DataFrame)
        assert isinstance(y, pd.DataFrame)
        y_hat = self.predict(x)

        roll_y = pd.Series(y.values.reshape(-1)).drop(['<PAD>', '*', '<STOP>', ','])
        roll_y_hat = pd.Series(y_hat.values.reshape(-1)).drop(['<PAD>', '*', '<STOP>', ','])

        index = pd.value_counts(y.values.reshape(-1)).index
        most_reacuent_tags = pd.Series(index, index=index).drop(['<PAD>', '<STOP>', '*'])[:10]
        sc = Score(most_reacuent_tags)
        sc.fit(roll_y, roll_y_hat)
        return sc.matrix_confusion()

    def accuracy(self, x, y):
        """

        :param x:
        :type x:
        :param y:
        :type y:
        :return:
        :rtype:
        """
        assert isinstance(x, pd.DataFrame)
        assert isinstance(y, pd.DataFrame)
        y_hat = self.predict(x)

        roll_y = pd.Series(y.values.reshape(-1)).drop(['<PAD>', '*', '<STOP>', ','])
        roll_y_hat = pd.Series(y_hat.values.reshape(-1)).drop(['<PAD>', '*', '<STOP>', ','])

        index = pd.value_counts(y.values.reshape(-1)).index
        most_reacuent_tags = pd.Series(index, index=index).drop(['<PAD>', '<STOP>', '*'])[:10]
        sc = Score(most_reacuent_tags)
        sc.fit(roll_y, roll_y_hat)
        return sc.over_all_acc()

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
        assert isinstance(sentence, FinkMos)
        y_1 = self.token2string[previous_tags[0]]
        y_2 = self.token2string[previous_tags[1]]
        y = self.token2string[next_tag]

        prop_q = sentence.prob_q(self.v, word_num, y, y_1, y_2)

        return prop_q

    def _viterbi(self, sentence):
        """
        :param sentence: ['*', '*', 'The', 'Treasury', 'is', ..., '<STOP>']
        :type sentence: pd.Series
        :param all_tags: tokenized tags
        :type all_tags: List TODO: np.array??
        :return: List of tags
        :rtype: List
        """
        num_words = len(sentence)  # includes '*','*' and <stop>
        sentence_fm = FinkMos(sentence, sentence, self.tests, self.tag_corpus)
        all_tags = self.tag_corpus_tokenized
        num_tags = len(all_tags)
        dims = (num_words, num_tags, num_tags)
        p_table = np.empty(dims, dtype=np.float)  # pi(k,u,v) - maximum probability of tag sequence

        # ending in tags u,v at position k
        # p_table[0, 0, 0] = 1  # init
        p_table[0, :, :] = 1  # init TODO:
        bp_table = np.empty(dims, dtype=np.int8)
        answer = [None] * num_words
        for k in range(1, num_words):
            print(str(k) + " out of " + str(num_words - 1))
            if k == 1:
                tags1 = [0]
            else:
                tags1 = all_tags
            for t1 in tags1:
                for t2 in all_tags:
                    if k in [1, 2]:  # 0 -> '*' tag
                        optional_tags = [0]
                    else:
                        optional_tags = all_tags
                    options = []  # np.array([])
                    for t_1 in optional_tags:  # t_1 is previous tag
                        # print("input_values: " + "t_1: " + str(t_1) + " t1 :" + str(t1) + " t2: " + str(t2))
                        options += [p_table[k - 1, t_1, t1] * self.model_function(next_tag=t2, word_num=k,
                                                                                  previous_tags=[t_1, t1],
                                                                                  sentence=sentence_fm)]
                    # print("num of options: " + str(len(np.unique(options))))
                    bp_table[k, t1, t2] = np.argmax(options)
                    p_table[k, t1, t2] = options[bp_table[k, t1, t2]]
        answer[num_words - 2], answer[num_words - 1] = np.unravel_index(bp_table[num_words - 1, :, :].argmax(),
                                                                        bp_table[num_words - 1, :, :].shape)  # argmax()

        # returns index of flatten array, unravel() gives the original 2d indices
        for k in reversed(range(num_words - 2)):
            answer[k] = bp_table[k + 2, answer[k + 1], answer[k + 2]]
        return answer

    def _translation(self):
        # assert (self.tag_corpus[0] == '*')
        self.token2string = {key: value for key, value in
                             enumerate(self.tag_corpus)}  # TODO make sure that self.tag_corpus[0] is '*'
        self.string2token = {value: key for key, value in enumerate(self.tag_corpus)}

    def _vectorize(self):

        a = FinkMos(self.x, self.y, tests=self.tests, tag_corpus=self.tag_corpus)
        a.fill_test()
        self.vector_x_y = a  # TODO change names
        self.lin_loss_matrix_x_y = a.f_x_y

    def _loss(self, v):
        positive = self._calculate_positive(v)
        non_linear = self._calculate_nonlinear(v)
        penalty = 0.1 * np.linalg.norm(v)

        return non_linear + penalty - positive

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
        return np.sum(self.vector_x_y.sentence_non_lineard_loss(v))
