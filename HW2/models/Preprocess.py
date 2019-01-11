# imports
from models.data_object import DP_sentence


#

class PreProcess:

    def __init__(self, path):
        self.path = path  # make sure the path is valid
        self.meta = dict()

    def parser(self):
        """
        :return: iterable of DP_sentence objects
        :rtype:
        """
        pass

    def get_stats(self):
        """

        :return: data stats
        :rtype:
        """
        pass
