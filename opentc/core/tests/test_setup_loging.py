import os
import logging
from unittest import TestCase
from opentc.util import setup_logging


class TestSetupLogging(TestCase):
    def test_logging_print(self):
        print("test")
        config_directories = [os.curdir, os.path.expanduser("~/.opentc"), "/etc/opentc", os.environ.get("OPENTC_CONF_DIR")]
        setup_logging(config_directories=config_directories)
        logger = logging.getLogger(__name__)
        logger.info("logging works")
        self.assertTrue(logging is not None)