from model import *
from features import *
from prerocesing import *
from score import *
from sentence_processor import *
from sentence_processor import *

import yaml

# Create features
with open(r"..\models\tests.YAML", 'r') as stream:
    tests_dict = yaml.load(stream)

tests = tests_dict['tests']
# tests = pass
# Load Data
data = PreprocessTags(True).load_data(
    r'..\data\train.wtag')

model1 = Model(tests)
# create f_x_y "matrix" for any x,y save indices of non zero tests TODO decide
a = model1.fit(data.x, data.y)

results = dict(
    test_sum=model1.lin_loss_matrix_x_y.sum().to_dict(),
    v=model1.v.tolist(),
    compare={test: dict(v_val=v_val, sum=sum) for test, v_val, sum in
             zip(tests, model1.v.tolist(), model1.lin_loss_matrix_x_y.sum().tolist())}
)
