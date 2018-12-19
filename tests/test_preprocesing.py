import unittest
from models.prerocesing import PreprocessTags

class test_PreprocessTags(unittest.TestCase):
    def test_Preprocess(self):
        a = PreprocessTags()
        line = "hola_a chau_b '_'"
        result = (['*', '*', 'hola', 'chau', "'", '<STOP>'],
                  ['*', '*', 'a', 'b', "'", '<STOP>'])
        self.failUnlessEqual(a._create_sentence(line), result)
        line = 'the_D dog_N barks_V ._.'
        result = (['*', '*', 'the', 'dog', 'barks', '<STOP>'],
                  ['*', '*', 'D', 'N', 'V', '<STOP>'])
        self.assertEqual(a._create_sentence(line), result)
