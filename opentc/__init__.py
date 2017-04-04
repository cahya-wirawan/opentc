"""
OpenTC is text classification engine running as client server architecture. It listen on port 3333

"""
import os
import logging.config
import yaml
import logging
from .version import __version__


__author__ = "Cahya Wirawan <Cahya.Wirawan@gmail.com>"


def setup_logging(
        config_directories=None,
        config_file=None,
        default_level=logging.INFO
):
    """Setup logging configuration

    """
    config_found = False
    config_file_path = None
    if config_file:
        config_file_path = config_file
        if os.path.isfile(config_file_path) and os.access(config_file_path, os.R_OK):
            config_found = True
    else:
        for directory in config_directories:
            if directory is None:
                continue
            config_file_path = os.path.join(directory, "logging.yml")
            if os.path.isfile(config_file_path) and os.access(config_file_path, os.R_OK):
                config_found = True
                break
    if config_found:
        with open(config_file_path, 'rt') as ymlfile:
            config = yaml.safe_load(ymlfile.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


def setup_config(
        config_directories=None,
        config_file=None,
        default_level=logging.INFO
):
    """Setup logging configuration

    """
    config_found = False
    config_file_path = None
    if config_file:
        config_file_path = config_file
        if os.path.isfile(config_file_path) and os.access(config_file_path, os.R_OK):
            config_found = True
    else:
        for directory in config_directories:
            if directory is None:
                continue
            config_file_path = os.path.join(directory, "opentc.yml")
            if os.path.isfile(config_file_path) and os.access(config_file_path, os.R_OK):
                config_found = True
                break
    if config_found:
        with open(config_file_path, 'rt') as ymlfile:
            config = yaml.safe_load(ymlfile.read())
        return config
    else:
        print("The configuration file is not found.")
        exit(1)
