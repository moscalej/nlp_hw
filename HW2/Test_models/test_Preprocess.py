"""
# Authors: Alejandro Moscoso <alejandro.moscoso@intel.com>
# 
"""
import unittest
from models.Preprocess import PreProcess
from models.data_object import DP_sentence

class test_Preprocess(unittest.TestCase):
    def test_init(self):
        PreProcess(r'../data/test.labeled')
    def test_parce(self):
        par = PreProcess(r'C:\technion\nlp_hw\HW2\data\test.labeled')
        par.parser()