"""
#
# Authors: Alejandro Moscoso <alejandro.moscoso@intel.com>
# 
"""

import time
import pandas as pd
import yaml
from HW2.models.boot_camp import Features
from HW2.models.dp_model import DP_Model
from HW2.models.Preprocess import PreProcess
from HW2.models.boot_camp import BootCamp
import numpy as np

with open(r'C:\Users\amoscoso\Documents\Technion\nlp\nlp_hw\HW2\local_paths.YAML') as f:
    paths = yaml.load(f)

NUM_EPOCHS = 10
MODELS = ['base', 'advance']
DATA_PATH = paths['train_data']  # 'toy_data'
TEST_PATH = paths['test_data']
RESULTS_PATH = paths['results_path']
WEIGHTS_PATH = paths['weights']
results_all = []

data = PreProcess(DATA_PATH).parser()
test = PreProcess(TEST_PATH).parser()
unlabel = PreProcess(paths['unlable_data'],label=False).parser()
# BASE MODEL
bc = BootCamp(Features('Advanse'))
model = DP_Model(boot_camp=bc)

start_time = time.time()
model.bc.investigate_soldiers(data)
model.bc.truncate_features(n_top=0, n_bottom=0)
total = model.bc.features.num_features
print(model.bc.features.num_features)

# %%
TRUNCATE_TOP = 0.05
TRUNCATE_BOT = 0.10
remove_top = int(total * TRUNCATE_TOP)
remove_bot = int(total * TRUNCATE_BOT)
model.bc.features.truncate_by_thresh(10_000,3)
# model.bc.truncate_features(n_top=remove_top, n_bottom=remove_bot)
print(model.bc.features.num_features)
# %%
model.bc.train_soldiers(data, fast=False)
# %%
model.w = np.zeros(model.bc.features.num_features)
#%%
result = model.perceptron(data, epochs=NUM_EPOCHS,validation=test)
result['epochs'] += 10 * len(results_all)
results_all.append(result)
#%%
df_results = pd.DataFrame(pd.concat(results_all))
df_results.to_csv(f'{RESULTS_PATH}\\re_{time.strftime("%d_%b_%y_%S_%M_%H")}.csv')
# model.w.tofile(f'{WEIGHTS_PATH}\\w_{time.strftime("%d_%b_%y_%S_%M_%H")}.h5')
