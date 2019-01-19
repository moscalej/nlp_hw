"""
#
# Authors: Alejandro Moscoso <alejandro.moscoso@intel.com>
# 
"""

import time

import pandas as pd
import yaml

from HW2.models.Preprocess import PreProcess
from HW2.models.boot_camp import BootCamp, Features
from HW2.models.dp_model import DP_Model

with open(r'local_paths.YAML') as f:
    paths = yaml.load(f)
toy_data = r'C:\Users\afinkels\Desktop\private\Technion\Master studies\עיבוד שפה טבעית\HW\hw_repo\nlp_hw\HW2\data\toy.labeled'
test_data = r'C:\Users\afinkels\Desktop\private\Technion\Master studies\עיבוד שפה טבעית\HW\hw_repo\nlp_hw\HW2\data\test.labeled'
train_data = r'C:\Users\afinkels\Desktop\private\Technion\Master studies\עיבוד שפה טבעית\HW\hw_repo\nlp_hw\HW2\data\train.labeled'
unlable_data = r'C:\Users\afinkels\Desktop\private\Technion\Master studies\עיבוד שפה טבעית\HW\hw_repo\nlp_hw\HW2\data\comp.unlabeled'
results_path = r'C:\Users\afinkels\Desktop\private\Technion\Master studies\עיבוד שפה טבעית\HW\hw_repo\nlp_hw\HW2\tests'
weights = r'C:\Users\afinkels\Desktop\private\Technion\Master studies\עיבוד שפה טבעית\HW\hw_repo\nlp_hw\HW2\tests\weights'
NUM_EPOCHS = [10]
MODELS = ['base', 'advance']
NUMBER_OF_FEATURES = [500, 5000, 50000, 100_000, 0]
DATA_PATH = toy_data
TEST_PATH = toy_data
RESULTS_PATH = results_path
WEIGHTS_PATH = results_path
results_all = []

data = PreProcess(DATA_PATH).parser()
test = PreProcess(TEST_PATH).parser()
# BASE MODEL
bc = BootCamp(Features('base'))
model = DP_Model(boot_camp=bc)
for n_epochs in NUM_EPOCHS:
    start_time = time.time()
    result = model.fit(data, epochs=n_epochs, truncate=100_000)
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
model.w.tofile(f'{WEIGHTS_PATH}\\w_{time.strftime("%d_%b_%y_%S_%M_%H")}.h5')
