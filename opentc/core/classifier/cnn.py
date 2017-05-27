from . cnn_text import CnnTextTraining, CnnTextEvaluator
from . import Classifier


class Cnn(Classifier):

    def __init__(self, cfg=None, categories=None, current_category=None, load=True):
        super().__init__()
        self.cfg = cfg
        self.categories = categories
        self.current_category = current_category
        if load:
            self.evaluator = CnnTextEvaluator(self.cfg, self.current_category)
        self.clf = None

    def fit(self, dataset, filename):
        self.logger.debug("train")
        self.clf = CnnTextTraining(self.cfg)
        self.clf.fit(dataset, filename)

    def reload(self, filename):
        self.logger.debug("reload")

    def predict(self, data):
        self.logger.debug("predict")
        result = self.evaluator.predict(data)
        result = ["{0}:{1:.2}".format(self.categories[result["predictions"][i]],
                                 result["probabilities"][i]) for i in range(len(result["predictions"]))]
        return result
