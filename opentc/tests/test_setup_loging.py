import logging
from unittest import TestCase
from opentc import setup_logging


class TestSetupLogging(TestCase):
    def test_logging_print(self):
        setup_logging()
        logger = logging.getLogger(__name__)
        logger.info("logging works")
        self.assertTrue(logging is not None)