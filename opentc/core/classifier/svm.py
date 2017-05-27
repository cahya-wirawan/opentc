import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib
from . import Classifier


class Svm(Classifier):

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
                             ('clf', SGDClassifier(loss='log', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)),
                             ])
        self.clf.fit(dataset.get_dataset()['data'], dataset.get_dataset()['target'])
        joblib.dump(self.clf, filename + ".pkl", compress=9)

    def reload(self, filename):
        self.logger.debug("reload")
        self.clf = joblib.load(filename)

    def predict(self, data):
        start = time.time()
        self.logger.debug("predict")
        prediction = self.clf.predict(data)
        probability = self.clf.predict_proba(data)
        result = ["{0}:{1:.2}".format(self.categories[prediction[i]], probability[i][prediction[i]])
                  for i in range(len(prediction))]
        end = time.time()
        self.logger.info("Predict time: {} seconds".format(end - start))
        return result
