"""
# Authors: Alejandro Moscoso <alejandro.moscoso@intel.com>
# 
"""
import unittest
from models.Preprocess import PreProcess
from models.data_object import DP_sentence

class test_Preprocess(unittest.TestCase):
    def test_init(self):
        # PreProcess(r'../data/test.labeled')
        PreProcess(
            r'C:\Users\afinkels\Desktop\private\Technion\Master studies\עיבוד שפה טבעית\HW\hw_repo\nlp_hw\HW2\data\test.labeled')
    def test_parce(self):
        par = PreProcess(r'../data/test.labeled')
        par = PreProcess(r'../data/test.labeled')
        par.parser()