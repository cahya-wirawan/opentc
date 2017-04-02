"""
OpenTC is text classification engine running as client server architecture. It listen on port 3333

"""
import os
import logging.config
import yaml
import logging


__author__ = "Cahya Wirawan <Cahya.Wirawan@gmail.com>"
__version__ = '0.2.0'


def setup_logging(
        default_path='logging.yml',
        default_level=logging.INFO,
        env_key='LOG_CFG'
):
    """Setup logging configuration

    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

