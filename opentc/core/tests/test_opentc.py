import json
import logging
import os
import tempfile
import time
import unittest

from opentc.util.client import Client

from opentc.core import __version__
from opentc.util import setup_logging
from opentc.core.server import Server


class TestOpentc(unittest.TestCase):
    config_directories = [os.curdir, os.path.expanduser("~/.opentc"), "/etc/opentc", os.environ.get("OPENTC_CONF_DIR")]
    setup_logging(config_directories=config_directories)
    logger = logging.getLogger(__name__)
    run_server = False
    address, port = "localhost", 3333
    server = None
    tcc = None
    data = b"Text Classification"*1000
    # md5sum = md5(data).hexdigest()
    md5sum = "4b1c78bb298ef3d3d3ee9a244cb5e0c6"
    x_raw = ["We would like to convert a group of these outlines into a 3D image. Someone mentioned that if we could \
                convert the TIFF into a vector format then we could view them in Autocad",
             "We do baptize converts, but no one who has been deceived into hearing the word is likely to be \
                a convert. If in fact the grace of God might work in such a situation, there is no harm done in \
                waiting a day or two",
             "Chronic persistent hepatitis is usually diagnosed when someone does a liver biopsy on a patient \
                that has persistently elevated serum transaminases months after a bout of acute viral hepatitis"]

    @classmethod
    def setUpClass(cls):
        TestOpentc.logger.debug("setUpClass")
        cls.tcc = Client(address=cls.address, port=cls.port)
        if cls.run_server:
            cls.server = Server()
            cls.server.start(run_forever=False, address=cls.address, port=cls.port)

    @classmethod
    def tearDownClass(cls):
        TestOpentc.logger.debug("tearDownClass")
        if cls.run_server:
            cls.server.shutdown()

    def setUp(self):
        self.logger.debug("setUp")

    def tearDown(self):
        self.logger.debug("tearDown")

    def test_ping(self):
        response = self.tcc.ping()
        response = json.loads(response.decode('utf-8'))
        self.logger.debug("{}: {}".format(self._testMethodName, response))
        self.assertEqual('PONG', response['result'])

    def test_version(self):
        response = self.tcc.command("VERSION\n")
        response = json.loads(response.decode('utf-8'))
        self.logger.debug("{}: {}".format(self._testMethodName, response))
        self.assertEqual(__version__, response['result'])

    def test_list_classifier(self):
        response = self.tcc.command("LIST_CLASSIFIER\n")
        response = json.loads(response.decode('utf-8'))
        self.logger.debug("{}: {}".format(self._testMethodName, response))
        self.assertEqual({'cnn': True, 'bayes': True, 'svm': True}, response['result'])

    def test_set_classifier(self):
        response = self.tcc.command("SET_CLASSIFIER:bayes:False\n")
        response = json.loads(response.decode('utf-8'))
        self.logger.debug("{}: {}".format(self._testMethodName, response))
        self.assertEqual([{'bayes': False}], response['result'])
        response = self.tcc.command("SET_CLASSIFIER:bayes:True\n")
        response = json.loads(response.decode('utf-8'))
        self.logger.debug("{}: {}".format(self._testMethodName, response))
        self.assertEqual([{'bayes': True}], response['result'])

    def test_md5_file(self):
        fd, temp_path = tempfile.mkstemp()
        file = os.fdopen(fd, "wb")
        file.write(self.data)
        file.close()
        response = self.tcc.md5_file(temp_path)
        response = json.loads(response.decode('utf-8'))
        os.remove(temp_path)
        self.logger.debug("{}: {}".format(self._testMethodName, response))
        self.assertEqual(self.md5sum, response['result'])

    def test_md5_stream(self):
        response = self.tcc.md5_stream(self.data)
        response = json.loads(response.decode('utf-8'))
        self.logger.debug("{}: {}".format(self._testMethodName, response))
        self.assertEqual(self.md5sum, response['result'])

    def test_predict_stream_0(self):
        start = time.time()
        response = self.tcc.predict_stream(self.x_raw[0].encode('utf-8'))
        self.assertIsNotNone(response)
        if response is not None:
            response = json.loads(response.decode('utf-8'))
            end = time.time()
            self.logger.debug("Time elapsed: {}".format(end - start))
            self.logger.debug("{}: {}".format(self._testMethodName, response))
            self.assertEqual({'bayes': ['comp.graphics'],
                              'cnn': ['comp.graphics'],
                              'svm': ['comp.graphics']}, response['result'])

    def test_predict_stream_1(self):
        start = time.time()
        response = self.tcc.predict_stream(self.x_raw[1].encode('utf-8'))
        self.assertIsNotNone(response)
        if response is not None:
            response = json.loads(response.decode('utf-8'))
            end = time.time()
            self.logger.debug("Time elapsed: {}".format(end - start))
            self.logger.debug("{}: {}".format(self._testMethodName, response))
            self.assertEqual({'bayes': ['soc.religion.christian'],
                              'cnn': ['soc.religion.christian'],
                              'svm': ['soc.religion.christian']}, response['result'])

    def test_predict_stream_2(self):
        start = time.time()
        response = self.tcc.predict_stream('\n'.join(self.x_raw).encode('utf-8'))
        self.assertIsNotNone(response)
        if response is not None:
            response = json.loads(response.decode('utf-8'))
            end = time.time()
            self.logger.debug("Time elapsed: {}".format(end - start))
            self.logger.debug("{}: {}".format(self._testMethodName, response))
            self.assertEqual({'bayes': ['comp.graphics', 'soc.religion.christian', 'sci.med'],
                              'cnn': ['comp.graphics', 'soc.religion.christian', 'sci.med'],
                              'svm': ['comp.graphics', 'soc.religion.christian', 'sci.med']},
                             response['result'])

    def test_predict_file(self):
        fd, temp_path = tempfile.mkstemp()
        file = os.fdopen(fd, "wb")
        file.write(self.x_raw[1].encode('utf-8'))
        file.close()
        response = self.tcc.predict_file(temp_path)
        response = json.loads(response.decode('utf-8'))
        os.remove(temp_path)
        self.logger.debug("{}: {}".format(self._testMethodName, response))
        self.assertEqual({'bayes': ['soc.religion.christian'],
                          'cnn': ['soc.religion.christian'],
                          'svm': ['soc.religion.christian']}, response['result'])

    def test_predict_file_multilines(self):
        fd, temp_path = tempfile.mkstemp()
        file = os.fdopen(fd, "wb")
        file.write('\n'.join(self.x_raw).encode('utf-8'))
        file.close()
        response = self.tcc.predict_file(temp_path)
        response = json.loads(response.decode('utf-8'))
        os.remove(temp_path)
        self.logger.debug("{}: {}".format(self._testMethodName, response))
        self.assertEqual({'bayes': ['comp.graphics', 'soc.religion.christian', 'sci.med'],
                          'cnn': ['comp.graphics', 'soc.religion.christian', 'sci.med'],
                          'svm': ['comp.graphics', 'soc.religion.christian', 'sci.med']},
                         response['result'])

    def test_unknown_command(self):
        response = self.tcc.command("Unknown command\n")
        response = json.loads(response.decode('utf-8'))
        self.logger.debug("{}: {}".format(self._testMethodName, response))
        self.assertEqual('Unknown Command', response['result'])

    def test_close(self):
        response = self.tcc.command("CLOSE\n")
        response = json.loads(response.decode('utf-8'))
        self.logger.debug("{}: {}".format(self._testMethodName, response))
        self.assertEqual('Bye', response['result'])
        TestOpentc.tcc = Client(address=self.address, port=self.port)

if __name__ == '__main__':
    unittest.main(verbosity=2)
