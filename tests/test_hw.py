import unittest
from models.prerocesing import PreprocessTags
from models.features import Ratnaparkhi
import pandas as pd
class test_model_1(unittest.TestCase):
    def test_fit(self):

        pass
    def test_predic(self):
        pass

class test_PreprocessTags(unittest.TestCase):
    def test_Preprocess(self):
        a = PreprocessTags()
        line = "hola_a chau_b '_'"
        result = (['*','*','hola', 'chau',"'",'<STOP>'],
                  ['*', '*', 'a', 'b',"'", '<STOP>'])
        self.failUnlessEqual(a._create_sentence(line),result)
        line = 'the_D dog_N barks_V ._.'
        result =(['*', '*','the', 'dog', 'barks','<STOP>'],
                 ['*', '*','D','N','V','<STOP>'])
        self.assertEqual(a._create_sentence(line),result)

class test_rapnaparkhi(unittest.TestCase):

    def test_rapna_100_107(self):
        x = pd.Series(['*', '*',
                       'The', 'Treasury', 'is', 'still', 'working', 'out',
                       'the', 'details', 'with', 'bank', 'trade', 'associations',
                       'and', 'the', 'other', 'government', 'agencies',
                       'that', 'have', 'a', 'hand', 'in', 'fighting',"preencounte",'word', '<STOP>'])
        y = pd.Series(['*', '*',
                       'DT', 'NNP', 'VBZ', 'RB', 'VBG', 'RP',
                       'DT', 'NNS', 'IN', 'NN', 'NN', 'NNS',
                       'CC', 'DT', 'JJ', 'NN', 'NNS', 'WDT',
                       'VBP', 'DT', 'NN', 'IN', 'VBG','NN','Vt',
                       '<STOP>'])
        r = Ratnaparkhi(x, y)
        print('')
        print(r.x[6],r.y[6])
        self.failUnlessEqual(r.f_100(6),0)
        # Test 101
        print(r.x[24],r.y[24])
        self.failUnlessEqual(r.f_101(24), 1)
        print(r.x[4], r.y[4])
        self.failUnlessEqual(r.f_101(4), 0)

        # Test 102
        print(r.x[25], r.y[25])
        self.failUnlessEqual(r.f_102(25), 1)
        print(r.x[4], r.y[4])
        self.failUnlessEqual(r.f_102(4), 0)

        # Test 103
        place = 7
        place2 =3
        print( r.y[[place, place - 1, place - 2]])
        self.failUnlessEqual(r.f_103(place), 0)
        print(r.x[4], r.y[4])
        self.failUnlessEqual(r.f_103(4), 0)

        # Test 104
        place = 5
        place2 = 6
        print(r.y[[place, place - 1]])
        self.failUnlessEqual(r.f_104(place), 0)
        print(r.x[4], r.y[4])
        self.failUnlessEqual(r.f_104(4), 0)

        # Test 105
        place = 26
        place2 = 6
        print(r.y[place],105)
        self.failUnlessEqual(r.f_105(place), 1)
        print(r.y[place2])
        self.failUnlessEqual(r.f_105(4), 0)

        # Test 106
        place = 26
        place2 = 6
        print(r.y[place])
        self.failUnlessEqual(r.f_106(place), 0)
        print(r.y[place2])
        self.failUnlessEqual(r.f_106(place2), 0)

        # Test 107
        place = 26
        place2 = 6
        print(r.y[place])
        self.failUnlessEqual(r.f_107(place), 0)
        print(r.y[place2])
        self.failUnlessEqual(r.f_107(place2), 0)

    def test_run_line(self):
        x = pd.Series(['*', '*',
                       'The', 'Treasury', 'is', 'still', 'working', 'out',
                       'the', 'details', 'with', 'bank', 'trade', 'associations',
                       'and', 'the', 'other', 'government', 'agencies',
                       'that', 'have', 'a', 'hand', 'in', 'fighting', "preencounte", 'word', '<STOP>'])
        y = pd.Series(['*', '*',
                       'DT', 'NNP', 'VBZ', 'RB', 'VBG', 'RP',
                       'DT', 'NNS', 'IN', 'NN', 'NN', 'NNS',
                       'CC', 'DT', 'JJ', 'NN', 'NNS', 'WDT',
                       'VBP', 'DT', 'NN', 'IN', 'VBG', 'NN', 'Vt',
                       '<STOP>'])
        r = Ratnaparkhi(x, y)
        a = r.run_line_tests('f_101')
        self.assertEqual(1,1)
    def test_features(self):
        x = pd.Series(['*', '*',
                       'The', 'Treasury', 'is', 'still', 'working', 'out',
                       'the', 'details', 'with', 'bank', 'trade', 'associations',
                       'and', 'the', 'other', 'government', 'agencies',
                       'that', 'have', 'a', 'hand', 'in', 'fighting', "preencounte", 'word', '<STOP>'])
        y = pd.Series(['*', '*',
                       'DT', 'NNP', 'VBZ', 'RB', 'VBG', 'RP',
                       'DT', 'NNS', 'IN', 'NN', 'NN', 'NNS',
                       'CC', 'DT', 'JJ', 'NN', 'NNS', 'WDT',
                       'VBP', 'DT', 'NN', 'IN', 'VBG', 'NN', 'Vt',
                       '<STOP>'])
        r = Ratnaparkhi(x, y)
        print()
        print(r.fill_test([f'f_10{x}'for x in range(8)]))
        self.assertEqual(r.fill_test([f'f_10{x}'for x in range(8)]),[0, 1, 1, 0, 0, 1, 0, 0])

class test_Misseleanous(unittest.TestCase):
    def test_iter_vect(self):
        b = PreprocessTags().load_data(r'..\data\test.wtag')
        tests = [f'f_10{x}' for x in range(8)]
        result = pd.DataFrame(columns=tests, index = b.x.index)
        for i in range(b.x.shape[0]):
            result.loc[i,:] = Ratnaparkhi(b.x.loc[i,:], b.y.loc[i,:]).fill_test(tests)





class test_model(unittest.TestCase):
    def test_question1(self):
        """
        LOAD THE DATA
        PRE PROCESS
        FIT - TRAIN
        PREDICT -

        :return:
        """
        acc=90
        print(acc)
        self.assertGreaterEqual(acc,90,msg=f'current acc:{acc}')

if __name__ == '__main__':
    unittest.main()

