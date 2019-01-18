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

NUM_EPOCHS = [10, 50, 100, 200]
MODELS = ['base', 'advance']
NUMBER_OF_FEATURES = [500, 5000, 50000, 100_000, 0]
DATA_PATH = r'C:\Users\afinkels\Desktop\private\Technion\Master studies\עיבוד שפה טבעית\HW\hw_repo\nlp_hw\HW2\data\toy.labeled'
TEST_PATH = r'C:\Users\afinkels\Desktop\private\Technion\Master studies\עיבוד שפה טבעית\HW\hw_repo\nlp_hw\HW2\data\toy.labeled'
RESULTS_PATH = r'C:\Users\afinkels\Desktop\private\Technion\Master studies\עיבוד שפה טבעית\HW\hw_repo\nlp_hw\HW2\Test_models'
results_all = []

data = PreProcess(DATA_PATH).parser()
test = PreProcess(TEST_PATH).parser()
# BASE MODEL
for n_epochs in NUM_EPOCHS:
    start_time = time.time()
    bc = BootCamp(Features('base'))
    model = DP_Model(boot_camp=bc)
    model.fit(data, epochs=n_epochs)
    train_acc = model.score(data)
    test_acc = model.score(test)
    results_all.append(['base', time.time() - start_time, n_epochs, train_acc, test_acc, bc.features.num_features])
    print(f'Finish base model with {n_epochs} epochs at {time.localtime()}')
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
#         print(f'Finish advance model with {n_epochs} epochs and {n_features} features at {time.localtime()}')
#
df_results = pd.DataFrame(results_all, columns=['Model', 'time', 'epochs', 'train_score', 'val_score', 'n_features'])
df_results.to_csv(f'{RESULTS_PATH}\\re_{time.localtime()}.csv')
