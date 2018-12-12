import unittest
from models.prerocesing import PreprocessTags
class test_model_1(unittest.TestCase):
    def test_fit(self):

        pass
    def test_predic(self):
        pass

class test_PreprocessTags(unittest.TestCase):
    def test_Preprocess(self):
        a = PreprocessTags(r'..\data\test.wtag')
        line = "hola_a chau_b '_'"
        result = (['*','*','hola', 'chau',"'",'<STOP>'],
                  ['*', '*', 'a', 'b',"'", '<STOP>'])
        self.assertEqual(a._create_sentence(line),result)
