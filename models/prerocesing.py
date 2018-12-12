import pandas as pd


class PreprocessTags:

    def __init__(self, path):
        x = []
        y = []
        with open(path, 'r') as fb:
            raw_file = fb.readlines()
        for line in raw_file:
            sentence, sentence_tag = self._create_sentence(line)
            x.append(sentence)
            y.append(sentence_tag)
        self.x = pd.DataFrame(x)
        self.y = pd.DataFrame(y)
        self.x.fillna('<PAD>', inplace=True)
        self.y.fillna('<PAD>', inplace=True)

    def _create_sentence(self, line):
        sentence = ['*', '*']
        sentence_tag = ['*', '*']
        line = line.strip(' ._.').split(' ')
        for word in line:
            temp = word.split('_')
            sentence.append(temp[0])
            sentence_tag.append(temp[1])
        sentence.append('<STOP>')
        sentence_tag.append('<STOP>')
        return sentence , sentence_tag


if __name__ == '__main__':
    a = PreprocessTags(r'D:\Ale\Documents\Technion\nlp\nlp_hw\data\train.wtag')
