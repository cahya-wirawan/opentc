from . import Dataset


class Mrpolarity(Dataset):

    def __init__(self, cfg=None):
        """
        Loads MR polarity data from files, splits the data into words and generates labels.
        Returns split sentences and labels.
        """
        super().__init__()
        # Load data from files
        positive_examples = []
        negative_examples = []
        try:
            positive_examples = list(open(cfg['positive_data_file']['path'], "r").readlines())
            positive_examples = [s.strip() for s in positive_examples]
            negative_examples = list(open(cfg['negative_data_file']['path'], "r").readlines())
            negative_examples = [s.strip() for s in negative_examples]
        except OSError as err:
            self.logger.error("OS error: {0}".format(err))
            exit(1)

        self.__dataset__ = dict()
        self.__dataset__['data'] = positive_examples + negative_examples
        target = [0 for x in positive_examples] + [1 for x in negative_examples]
        self.__dataset__['target'] = target
        self.__dataset__['target_names'] = ['positive_examples', 'negative_examples']
