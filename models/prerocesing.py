import pandas as pd


class PreprocessTags:

    def __init__(self, optimize=False):
        self.x = pd.DataFrame()
        self.y = pd.DataFrame()
        self.path = str()
        self.optimize = optimize

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
        if self.optimize == True:
            self._optimize()
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
            if len(temp) == 2:
                sentence.append(temp[0])
                sentence_tag.append(temp[1])
        sentence.append('<STOP>')
        sentence_tag.append('<STOP>')
        return sentence, sentence_tag

    def _optimize(self):
        self.x = self.x.apply(self.optimizer)
        t = self.x.value_counts(ascending=False)
        a = pd.DataFrame([t.tolist(), t.index.tolist()]).T
        b = a[a[0] < 5]
        tt = {word: "<" for word in b[1]}
        self.x = self.x.apply(lambda x: tt[x] if x in tt else x)

    def optimizer(self, word):
        change = False
        newword = "<"
        if len(word) < 4:
            return word
        if word[:3] == 'pre':
            newword = "pre" + "<"
            change = True
        if word[-3:] == 'ing':
            newword += 'ing'
            change = True

        return newword if change else word



if __name__ == '__main__':
    # a = PreprocessTags().load_comp(r'D:\Ale\Documents\Technion\nlp\nlp_hw\data\comp.words')
    b = PreprocessTags(True).load_data(r'D:\Ale\Documents\Technion\nlp\nlp_hw\data\train.wtag')
