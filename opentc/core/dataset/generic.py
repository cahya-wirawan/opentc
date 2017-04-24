from sklearn.datasets import load_files
from . import Dataset


class Generic(Dataset):

    def __init__(self, cfg=None):
        """
        Load text files with categories as subfolder names.
        Individual samples are assumed to be files stored a two levels folder structure.
        :param container_path: The path of the container
        :param categories: List of classes to choose, all classes are chosen by default (if empty or omitted)
        :param shuffle: shuffle the list or not
        :param random_state: seed integer to shuffle the dataset
        :return: data and labels of the dataset
        """
        super().__init__()
        self.__dataset__ = load_files(container_path=cfg['container_path'], categories=cfg['categories'],
                                      load_content=cfg['load_content'], shuffle=cfg['shuffle'],
                                      encoding=cfg['encoding'], random_state=cfg['random_state'])
