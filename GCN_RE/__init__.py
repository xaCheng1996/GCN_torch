import numpy as np
import pickle
import random
import sys
import logging

# import GCN_RE.utils as utils
from GCN_RE.GCN_model import GCNReModel
import GCN_RE.utils as utils


class GCNRE:
    _logger = logging.getLogger(__name__)
    def test(self, RE_filename, dataset, data_name, threshold):
        '''
        The system tests the current NER model against a text in the CONLL format.

        :param dataset: the filename of a text in the CONLL format
        :return: None, the function prints precision, recall and chunck F1
        '''
        sentences = utils.auxiliary.get_all_sentence(dataset, data_name)
        maxlength = 256
        data = utils.auxiliary.get_data_from_sentences(sentences, maxlength)
        self.model = GCNReModel(maxlength=maxlength, dropout=1.0)
        self.model.load_tensorflow(RE_filename)
        precision, recall ,f1 = utils.testing.get_gcn_results(self.model, data, maxlength, threshold=threshold)
        print('precision:', precision)
        print('recall:', recall)
        print('F1:', f1)

    @staticmethod
    def train_and_save(dataset, saving_dir, data_name, epochs, bucket_size):
        '''
        :param dataset: A file use as a training.
        :param saving_dir: The directory where to save the results
        :param epochs: The number of epochs to use in the training
        :param bucket_size: The batch size of the training.
        :return: An instance of this class
        '''
        print('Training the system according to the dataset ', dataset)
        return utils.training.train_and_save(dataset, saving_dir, data_name, epochs, bucket_size)
