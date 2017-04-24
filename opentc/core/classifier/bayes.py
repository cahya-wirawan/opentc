from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from . import Classifier


class Bayes(Classifier):

    def __init__(self, cfg=None, categories=None, current_category=None, load=True):
        super().__init__()
        self.categories = categories
        self.current_category = current_category
        if load:
            self.clf = joblib.load(cfg['pre_trained_file'][self.current_category])

    def fit(self, dataset, filename):
        self.logger.debug("fit")
        self.clf = Pipeline([('vect', CountVectorizer()),
                             ('tfidf', TfidfTransformer()),
                             ('clf', MultinomialNB())
                             ])
        self.clf.fit(dataset.get_dataset()['data'], dataset.get_dataset()['target'])
        joblib.dump(self.clf, filename + ".pkl", compress=9)

    def reload(self, filename):
        self.logger.debug("reload")
        self.clf = joblib.load(filename)

    def predict(self, data):
        self.logger.debug("predict")
        predicted = self.clf.predict(data)
        predicted = [self.categories[i] for i in predicted]
        return predicted
