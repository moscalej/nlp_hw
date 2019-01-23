"""
# Authors: Alejandro Moscoso <alejandro.moscoso@intel.com>
# 
"""

import unittest
import yaml
from HW2.models.Preprocess import PreProcess
from HW2.models.data_object import DP_sentence
with open(r'C:\Users\amoscoso\Documents\Technion\nlp\nlp_hw\HW2\local_paths.YAML') as f:
    paths = yaml.load(f)

class test_Preprocess(unittest.TestCase):
    def test_init(self):
        # PreProcess(r'../data/test.labeled')
        PreProcess(
            r'C:\Users\afinkels\Desktop\private\Technion\Master studies\עיבוד שפה טבעית\HW\hw_repo\nlp_hw\HW2\data\test.labeled')
    def test_parce(self):
        par = PreProcess(paths['toy_data'])
        obj = par.parser()
        par._so2df(obj[0])