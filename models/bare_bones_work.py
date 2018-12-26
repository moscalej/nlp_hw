import numpy as np

matrix = np.load(r'D:\Ale\Documents\Technion\nlp\nlp_hw\tests\250_tests.h5')
v = np.ones([250, 1]) * 3


def minimize_func(v):
    F_V = matrix @ v  # add factor
    F_V_mask = F_V  # * mask
    L_FV = np.sum(np.sum(F_V_mask))  # * mask
    exp_ = np.exp(F_V)
    sum_exp = np.sum(exp_, axis=0)  # * np.sum(mask,axis=0)# mask ??
    ln = np.log(sum_exp)
    sum_ln = np.sum(ln)
    return sum_ln - L_FV  # todo add regularization


def q_table():
    F_V = matrix @ v_opmily
    exp_ = np.exp(F_V)
    self.qtable = F_V / (
        np.sum(exp_, axis=0))  # out is a matrix where [tuples * tag] denomitor vector for sofmax  element per tuple
