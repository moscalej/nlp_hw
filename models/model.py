



class Model:
    def __init__(self, tests=[]):
        self.test = tests
        self.v = None

    def fit(self,x , y, learning_rate=0.02,x_val=None , y_val = None):
        """
        Fit will train the Model
            - Encoding
                - For the data will create a Tensor [ # todo Read more
            - Gradient decent
                -Loss
                - update V
            - Calculate metrics

        :param x: DataFrame [row = sentence , col = word]
        :param y: DataFrame [row = sentence tag , col = Word tag]
        :param learning_rate: [don't know if need it] # TODO check if remove
        :param x_val:[row = sentence , col = word]
        :param y_val:[row = sentence tag , col = Word tag]
        :return: metrics dict {} $# TODO check witch metrics we need
        """
        pass

    def predict(self,x):
        """
        This will work with the Viterbi
        :param x:
        :return:
        """
        pass

