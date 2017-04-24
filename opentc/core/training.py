
import logging
import logging.config
import os
from datetime import datetime

from opentc.core.dataset import Dataset


class Training(object):
    """
    Class for using OpenTCServer with a network socket
    """
    classifiers = dict()

    def __init__(self, cfg):
        """
        class initialisation
        """
        self.logger = logging.getLogger(__name__)
        self.cfg = cfg

        try:
            os.makedirs(self.cfg["result_dir"], exist_ok=True)
        except OSError as err:
            self.logger.error("OS error: {0}".format(err))

        for classifier_name in self.cfg["classifiers"]:
            if classifier_name == "default":
                continue
            module = __import__("opentc.core.classifier." + classifier_name)
            class_ = getattr(getattr(getattr(getattr(module, "core"), "classifier"),
                                     classifier_name), classifier_name.title())
            if class_:
                classifier = dict()
                classifier["enabled"] = self.cfg["classifiers"][classifier_name]["enabled"]
                default_dataset = self.cfg["datasets"]["default"]
                classifier["class"] = class_(self.cfg["classifiers"][classifier_name],
                                             self.cfg["datasets"][default_dataset]["categories"],
                                             default_dataset, False)
                Training.classifiers[classifier_name] = classifier

    def start(self, cn=None, dn=None):
        self.logger.info("Training starts")
        if cn:
            classifier_name = cn
        else:
            classifier_name = self.cfg["classifiers"]["default"]
        if classifier_name not in Training.classifiers.keys() and classifier_name != "all":
            print("The classifier {} doesn't exist".format(classifier_name))
            return 1
        if dn:
            dataset_name = dn
        else:
            dataset_name = self.cfg["datasets"]["default"]
        if dataset_name not in self.cfg["datasets"]:
            print("The dataset {} doesn't exist".format(dataset_name))
            return 1
        dataset = Dataset.create_dataset(self.cfg["datasets"][dataset_name])
        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        training = Training
        if classifier_name == "all":
            for classifier_name in training.classifiers:
                result_name = "{}/{}_{}_{}".format(self.cfg["result_dir"], classifier_name, dataset_name, now)
                training.classifiers[classifier_name]["class"].fit(dataset, result_name)
                print("The training of {} classifier for the dataset {} is done.".format(classifier_name, dataset_name))
                print("The result is saved in: {}(.pkl)".format(result_name))
        else:
            result_name = "{}/{}_{}_{}".format(self.cfg["result_dir"], classifier_name, dataset_name, now)
            training.classifiers[classifier_name]["class"].fit(dataset, result_name)
            print("The training of {} classifier for the dataset {} is done.".format(classifier_name, dataset_name))
            print("The result is saved in: {}(.pkl)".format(result_name))

        self.logger.info("Training end")
