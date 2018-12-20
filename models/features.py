class Features:
    def get_tests(self):
        return dict(
            f_100=self.f_100,
            f_101=self.f_101,
            f_102=self.f_102,
            f_103=self.f_103,
            f_104=self.f_104,
            f_105=self.f_105,
            f_106=self.f_106,
            f_107=self.f_107,

            oun_000= lambda sentence, place, y, y_1, y_2:
            1 if sentence[place] =='the' and y =='IN' else 0,

            oun_001=lambda sentence, place, y, y_1, y_2:
            1 if sentence[place] == 'about' and y == 'IN' else 0,

            oun_002=lambda sentence, place, y, y_1, y_2:
            1 if y_1 == 'IN' and y == 'JJ' else 0,

            oun_003=lambda sentence, place, y, y_1, y_2:
            1 if y_1 == 'IN'and y_2 == 'NNS' and y == 'JJ' else 0,

            oun_004=lambda sentence, place, y, y_1, y_2:
            1 if sentence[place] == 'the' and y == 'IN' else 0,

            oun_005=lambda sentence, place, y, y_1, y_2:
            1 if sentence[place] == 'the' and y == 'IN' else 0,
        )

    def f_100(self, sentence, place, y, y_1, y_2):
        return 1 if sentence[place] == 'base' and y == 'Vt' else 0

    def f_101(self, sentence, place, y, y_1, y_2):
        if len(sentence[place]) < 4:
            return 0
        return 1 if sentence[place][-3:] == 'ing' and y == 'VBG' else 0

    def f_102(self, sentence, place, y, y_1, y_2):
        if len(sentence[place]) < 4:
            return 0
        return 1 if sentence[place][:3] == 'pre' and y == 'NN' else 0

    def f_103(self, sentence, place, y, y_1, y_2):
        return 1 if y == 'Vt' and y_1 == 'JJ' and y_2 == 'DT' else 0

    def f_104(self, sentence, place, y, y_1, y_2):
        return 1 if y == 'Vt' and y_1 == 'JJ' else 0

    def f_105(self, sentence, place, y, y_1, y_2):
        return 1 if y == 'Vt' else 0

    def f_106(self, sentence, place, y, y_1, y_2):
        return 1 if sentence[place - 1] == 'the' and y == 'Vt' else 0

    def f_107(self, sentence, place, y, y_1, y_2):
        if place == sentence.size - 1:
            return 0
        return 1 if sentence[place + 1] == 'the' and y == 'Vt' else 0
