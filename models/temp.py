import numpy as np


def viterbi(model, v, sentence, all_tags):
    word_num = len(sentence)
    # init
    table_dims = 1  # TODO
    p_table = np.array(ndim=3)  # pi(k,u,v) - maximum probability of tag sequence
    # ending in tags u,v at position k
    # init pi(0,*,*)=1
    p_table[0,0,0] = 1
    bp_table = np.array(ndim=3)
    #
    answer = [None] * word_num
    for k in range(word_num):
        if k == 1:
            tags1 = [0]
        elif k < len(sentence) - 2:
            tags1 = all_tags
        for t1 in range(len(tags1)):
            for t2 in range(len(all_tags)):
                if k == 1 or k == 2:  # 0 -> '*' tag
                    p_table[k, t1, t2] = p_table[k - 1, 0, t1] * model.eval(tag=t2, word_num=k,
                                                                                 previous_tags=[0, t1],
                                                                                 sentence=sentence,
                                                                                 weights=v)
                    bp_table[k, t1, t2] = 0
                else:
                    options = p_table[k - 1, all_tags, t1] * model.eval(tag=t2, word_num=k,
                                                                                 previous_tags=[range(all_tags), t1],
                                                                                 sentence=sentence,
                                                                                 weights=v)
                    bp_table[k, t1, t2] = np.argmax(options)
                    p_table[k, t1, t2] = options[bp_table[k, t1, t2]]

    answer[word_num-1], answer[word_num] = np.argmax(p_table[word_num,:,:])
    for k in reversed(range(word_num-2)):
        answer[k] = bp_table[k+2,answer[k+1],answer[k+2]]

    return answer



def q(v, t, senteces):
    for
