# imports
from HW2.models.data_object import DP_sentence
import pandas as pd
import numpy as np


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
        all_data = pd.read_csv(self.path,
                               names=['TC', 'TOKEN', 'n1', 'TP', 'n2', 'n3', 'TH', 'DL', 'n4', 'n5'],
                               delim_whitespace=True)
        all_data = all_data.filter(items=['TC', 'TOKEN', 'TP', 'TH', 'DL'], axis=1)
        all_data['SN'] = self._sente_number(all_data['TC'])
        groups = all_data.groupby('SN')
        sentence_objects = []
        for group in groups:
            sentence_objects.append(self._df2so(group))  # Data frame to Sentence object
        return sentence_objects

    def _sente_number(self, series):
        values = []
        sentence_number = 0
        for val in series:
            if val == 1:
                sentence_number += 1
            values.append(sentence_number)
        return np.array(values)

    def get_stats(self):
        """

        :return: data stats
        :rtype:
        """
        pass

    def _df2so(self, group):
        (number, df) = group
        # assert (isinstance(group,pd.DataFrame))
        x = np.concatenate((['<ROOT>'], df['TOKEN'].values))
        tags = np.concatenate((['<ROOT>'], df['TP'].values))
        graph_dict = {key: [] for key in df['TH'].unique()}
        for tc, th in zip(df['TC'], df['TH']):
            graph_dict[th].append(tc)
        return DP_sentence(sentence=x,
                           tags=tags,
                           graph=graph_dict)

    def from_ds_to_file(self, iter_ds):

        pass

    def _so2df(self, so):  # Sentence Object
        # assert isinstance(so,DP_sentence)
        tags = so.tags
        th = np.zeros(so.sentence.shape[0])
        tc = [x for x in range(1, so.sentence.shape[0] + 1)]
        for key, value in so.tags:
            th[value] = key
        results = pd.DataFrame(columns=['TC', 'TOKEN', 'n1', 'TP', 'n2', 'n3', 'TH', 'DL', 'n4', 'n5'])
        results['TC'] = tc
        results['TOKEN'] = so.sentence
        results['TP'] = so.tags
        results['TH'] = th
        results.fillna('_')
        return results
