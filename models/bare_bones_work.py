#
# matrix = np.load(r'D:\Ale\Documents\Technion\nlp\nlp_hw\tests\250_tests.h5')
# v = np.ones([250, 1]) * 3
import numpy as np
import pandas as pd

from prerocesing import PreprocessTags
from sentence_processor import FinkMos


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


data = PreprocessTags(True).load_data(
    r'..\data\toy_dataset.txt')
tag_corp = pd.Series(data.y[0:10000]).unique()
fm = FinkMos(data.x[0:10000], data.y[0:10000], tag_corp)
fm.create_tuples()
print("fm.weight_mat")
print(fm.weight_mat)
print("fm.tuple_5_list")
print(fm.tuple_5_list)
fm.create_feature_sparse_list_v2()
print(len(fm.f_matrix_list))
print(fm.f_matrix_list[0].shape)
fm.minimize_loss()
