import numpy as np


def viterbi(model, sentence, all_tags):
    """
    :param model: Used model for learning
    :type model: Class model
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
    p_table = np.empty(dims, dtype=np.int8)  # pi(k,u,v) - maximum probability of tag sequence
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
                options = p_table[k - 1, w, t1] * model.eval(next_tag=t2, word_num=k,
                                                             previous_tags=[w, t1],
                                                             sentence=sentence)
                bp_table[k, t1, t2] = np.argmax(options)
                print("bp_table[k, t1, t2]")
                print(bp_table[k, t1, t2])
                p_table[k, t1, t2] = options[bp_table[k, t1, t2]]

    answer[num_words - 1], answer[num_words] = np.argmax(bp_table[num_words, :, :])
    for k in reversed(range(num_words - 2)):
        answer[k] = bp_table[k + 2, answer[k + 1], answer[k + 2]]

    return answer


# Testing
class DummyModel:

    def __init__(self, weights, q_func, feature_factory):
        self.weights = weights
        self.q = q_func
        self.feature_factory = feature_factory

    def eval(self, next_tag, word_num, previous_tags, sentence):
        q_val = self.q(next_tag, word_num, previous_tags, sentence, self.feature_factory)
        res = np.dot(self.weights, q_val)
        return res


def q_func(next_tag, word_num, previous_tags, sentence, feature_factory):
    ans = feature_factory(previous_tags, word_num, sentence, next_tag)
    return previous_tags[0]


def feature_factory(previous_tags, word_num, sentence, tag):
    feature_num = 10
    return 10 * [(tag % 2)]


if __name__ == '__main__':
    weights = 10
    feature_num = 10
    num_tags = 10
    weights = np.random.rand(feature_num)
    model = DummyModel(weights, q_func)
    result = viterbi(model, sentence=["the", "dog", "barks"], all_tags=range(num_tags))
