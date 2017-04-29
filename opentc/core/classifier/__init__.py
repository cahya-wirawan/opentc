import logging
from abc import ABCMeta, abstractmethod

ABC = ABCMeta('ABC', (object,), {})


class Classifier(ABC):
    __instances__ = dict()

    def __init__(self):
        self.logger = logging.getLogger(self.__module__)
        Classifier.__instances__[self.__class__.__name__] = self

    @abstractmethod
    def fit(self, dataset, filename):
        pass

    @abstractmethod
    def reload(self, filename):
        pass

    @abstractmethod
    def predict(self, data):
        pass