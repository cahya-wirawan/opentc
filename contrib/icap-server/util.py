#!/bin/env python

import os
import logging.config
import yaml


def setup_logging(
        config_directories=None,
        config_file=None,
        default_level=logging.INFO,
        default_filename="logging.yml"
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
            config_file_path = os.path.join(directory, default_filename)
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
        default_filename="icap-server-opentc.yml"
):
    """Setup configuration

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
            config_file_path = os.path.join(directory, default_filename)
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
