import unittest

import numpy as np
import pandas as pd
from models.model import Model
import models.features as feat
from models.features import Features
from models.prerocesing import PreprocessTags
from models.sentence_processor import FinkMos
import os

os.chdir(r'C:\Users\amoscoso\Documents\Technion\nlp\nlp_hw\tests')
# %%
data = PreprocessTags(True).load_data(
    r'..\data\train.wtag')
word_num = 500
# generate tests - (comment out if file is updated)
feat_generator = Features()
feat_generator.generate_tuple_corpus(data.x[0:word_num], data.y[0:word_num])
for template in feat.templates_dict.values():
    feat_generator.generate_lambdas(template['func'], template['tuples'])
feat_generator.save_tests()
test_data = PreprocessTags(True).load_data(
    r'..\data\test.wtag')
# %%
word_num = 500
test_number = 50
model1 = Model()
model1.fit(data.x[0:word_num], data.y[0:word_num])

y_hat = model1.predict(test_data.x[:test_number])
model1.confusion(y_hat, data.y[:test_number])
