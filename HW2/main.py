"""
#
# Authors: Alejandro Moscoso <alejandro.moscoso@intel.com>
# 
"""

from HW2.models.boot_camp import BootCamp, Features
from HW2.models.Preprocess import PreProcess
from HW2.models.data_object import DP_sentence
from HW2.models.dp_model import DP_Model
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

with open(r'C:\technion\nlp_hw\HW2\local_paths.YAML') as f:
    paths = yaml.load(f)

NUM_EPOCHS = [10]
MODELS = ['base', 'advance']
NUMBER_OF_FEATURES = [500, 5000, 50000, 100_000, 0]
DATA_PATH = paths['test_data']
TEST_PATH = paths['toy_data']
RESULTS_PATH = paths['results_path']
results_all = []

data = PreProcess(DATA_PATH).parser()
test = PreProcess(TEST_PATH).parser()
# BASE MODEL
bc = BootCamp(Features('base'))
model = DP_Model(boot_camp=bc)
for n_epochs in NUM_EPOCHS:
    start_time = time.time()
    result = model.fit(data, epochs=n_epochs)
    results_all.append(result)
#
# # Advance
# for n_epochs in NUM_EPOCHS:
#     for n_features in NUMBER_OF_FEATURES:
#         start_time = time.time()
#         bc = BootCamp(Features('base'))
#         model = DP_Model(boot_camp=bc)
#         model.fit(data, epochs=n_epochs, truncate=0)
#         train_acc = model.score(data)
#         test_acc = model.score(test)
#         results_all.append(['base', time.time() - start_time, n_epochs, train_acc, test_acc, n_features])
#         print(f'Finish advance model with {n_epochs} epochs and {n_features} features at {time.strftime("%X %x")}')
#
df_results = pd.DataFrame(pd.concat(results_all))
df_results.to_csv(f'{RESULTS_PATH}\\re_{time.strftime("%d_%b_%y_%S_%M_%H")}.csv')
