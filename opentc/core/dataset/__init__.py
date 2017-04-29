import numpy as np
import re
import logging
from abc import ABCMeta

ABC = ABCMeta('ABC', (object,), {})


class Dataset(ABC):
    __dataset__ = None
    __instances__ = dict()

    def __init__(self):
        self.logger = logging.getLogger(self.__module__)
        Dataset.__instances__[self.__class__.__name__] = self

    @staticmethod
    def clean_string(string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def load_data_labels(self):
        """
        Load data and labels
        :return: data and labels
        """
        self.logger.debug("load_data_labels")
        # Split by words
        x_text = self.__dataset__['data']
        x_text = [Dataset.clean_string(sent) for sent in x_text]
        # Generate labels
        labels = []
        for i in range(len(x_text)):
            label = [0 for j in self.__dataset__['target_names']]
            label[self.__dataset__['target'][i]] = 1
            labels.append(label)
        y = np.array(labels)
        return [x_text, y]

    def get_dataset(self):
        return self.__dataset__

    @staticmethod
    def create_dataset(dataset):
        module = __import__("opentc.core.dataset." + dataset['name'])
        class_ = getattr(getattr(getattr(getattr(module, "core"), "dataset"), dataset['name']), dataset['name'].title().replace("_",""))
        instance = class_(dataset)
        return instance
