import pandas as pd


class PreprocessTags:

    def __init__(self):
        self.x = pd.DataFrame()
        self.y = pd.DataFrame()
        self.path = str()

    def load_data(self, path):
        x = []
        y = []
        with open(path, 'r') as fb:
            raw_file = fb.readlines()
        for line in raw_file:
            sentence, sentence_tag = self._create_sentence(line)
            x+=sentence
            y+=sentence_tag
        self.x = pd.Series(x)
        self.y = pd.Series(y)
        return self

    def load_comp(self, path):
        x = []
        with open(path, 'r') as fb:
            raw_file = fb.readlines()
        for line in raw_file:
            temp = line.strip().strip(" .")
            sentence = ['*', '*'] + temp.split()

            sentence.append('<STOP>')
            x.append(*sentence)
        self.x = pd.Series(x)
        return self

    def _create_sentence(self, line):
        sentence = ['*', '*']
        sentence_tag = ['*', '*']
        line = line.strip().strip('? ._.').split(' ')
        for word in line:
            temp = word.split('_')
            sentence.append(temp[0])
            sentence_tag.append(temp[1])
        sentence.append('<STOP>')
        sentence_tag.append('<STOP>')
        return sentence, sentence_tag


if __name__ == '__main__':
    a = PreprocessTags().load_comp(r'D:\Ale\Documents\Technion\nlp\nlp_hw\data\comp.words')
    b = PreprocessTags().load_data(r'D:\Ale\Documents\Technion\nlp\nlp_hw\data\test.wtag')
