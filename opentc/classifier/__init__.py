import logging
from .. import setup_logging
from abc import ABC, abstractmethod


class Classifier(ABC):
    __instances__ = dict()

    def __init__(self):
        setup_logging()
        self.logger = logging.getLogger(__name__)
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