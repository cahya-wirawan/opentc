import sys
import logging.config
import threading
import socketserver
import socket
import hashlib
import struct
import json
import logging
from opentc.core import __version__


class Server(object):
    """
    Class for using Server with a network socket
    """
    classifiers = dict()
    port = None

    def __init__(self, cfg=None):
        """
        class initialisation
        """
        self.logger = logging.getLogger(__name__)
        self.cfg = cfg
        self.server = None
        self.address = None
        self.port = None
        self.timeout = None
        for classifier_name in self.cfg['classifiers']:
            if classifier_name == "default":
                continue
            module = __import__("opentc.core.classifier." + classifier_name)
            class_ = getattr(getattr(getattr(getattr(module, "core"), "classifier"),
                                     classifier_name), classifier_name.title())
            # class_ = attr4
            if class_ is not None:
                classifier = dict()
                classifier['enabled'] = self.cfg['classifiers'][classifier_name]['enabled']
                dataset_name = self.cfg['dataset']['name']
                classifier['class'] = class_(self.cfg['classifiers'][classifier_name],
                                             self.cfg['dataset']['categories'],
                                             dataset_name)
                Server.classifiers[classifier_name] = classifier

    def start(self, address=None, port=None, timeout=None, run_forever=True):
        """
        :param run_forever:
        :param address: hostname or ip address
        :param port: TCP port
        :param timeout: socket timeout
        :return:
        """
        if address:
            self.address = address
        else:
            self.address = self.cfg["address"]
        if port:
            self.port = port
        else:
            self.port = self.cfg["port"]
        if timeout:
            self.timeout = timeout
        else:
            self.timeout = self.cfg["timeout"]

        try:
            self.server = self.ThreadedTCPServer((self.address, self.port),
                                                 self.ThreadedTCPRequestHandler)
            self.server.socket.settimeout(self.timeout)
            self.server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            # Start a thread with the server -- that thread will then start one
            # more thread for each request
            server_thread = threading.Thread(target=self.server.serve_forever)
            # Exit the server thread when the main thread terminates
            server_thread.daemon = True
            server_thread.start()
            self.logger.info("Server loop running in thread: {}".format(server_thread.name))
            if run_forever:
                self.server.serve_forever()
        except socket.error:
            e = sys.exc_info()[1]
            raise ConnectionError(e)

    def shutdown(self):
        self.logger.info("Server shutdown")
        self.server.shutdown()

    class ThreadedTCPRequestHandler(socketserver.StreamRequestHandler):
        max_buffer_size = 4096

        def __init__(self, request, client_address, server):
            self.logger = logging.getLogger(__name__)
            super().__init__(request, client_address, server)

        def handle(self):

            try:
                cur_thread = threading.current_thread()
                while True:
                    data = self.receive()
                    if data is None:
                        break
                    data = data.rstrip()
                    self.logger.info("Thread {} received: {}".format(cur_thread.name, data))
                    header = data.split(b':')
                    if header[0] == b'PING':
                        self.ping(header[1].decode('utf-8'))
                    elif header[0] == b'VERSION':
                        self.version()
                    elif header[0] == b'RELOAD':
                        self.reload()
                    elif header[0] == b'LIST_CLASSIFIER':
                        self.list_classifier()
                    elif header[0] == b'SET_CLASSIFIER':
                        classifier = header[1].decode('utf-8')
                        value = header[2].decode('utf-8')
                        self.set_classifier(classifier, value)
                    elif header[0] == b'MD5_FILE':
                        file_name = header[1]
                        self.md5_file(file_name=file_name)
                    elif header[0] == b'MD5_STREAM':
                        self.md5_stream()
                    elif header[0] == b'PREDICT_STREAM':
                        self.predict_stream(header[1].decode('utf-8'))
                    elif header[0] == b'PREDICT_FILE':
                        mid = header[1].decode('utf-8')
                        file_name = header[2]
                        self.predict_file(mid=mid, file_name=file_name)
                    elif header[0] == b'CLOSE':
                        self.close()
                        break
                    else:
                        self.unknown_command()
            except socket.error:
                e = sys.exc_info()[1]
                raise ConnectionError(e)
            self.logger.info("Thread {} exit".format(cur_thread.name))

        def send(self, data):
            size = len(data)
            packed_header = struct.pack('=I', size)
            self.request.sendall(packed_header + data)

        def receive(self):
            packed_header = self.rfile.read(4)
            if packed_header == b'':
                return None
            (size, ) = struct.unpack('=I', packed_header)
            if size == 0 or size > self.max_buffer_size:
                return None
            data = self.rfile.read(size)
            return data

        def ping(self, mid=None):
            response = dict()
            response["status"] = "OK"
            response["mid"] = mid
            response["result"] = "PONG"
            response = json.dumps(response).encode('utf-8')
            self.send(response)
            self.logger.debug("Ping sent pong with mid: {}".format(mid))

        def close(self):
            response = dict()
            response["status"] = "OK"
            response["result"] = "Bye"
            response = json.dumps(response).encode('utf-8')
            self.send(response)

        def version(self):
            response = dict()
            response["status"] = "OK"
            response["result"] = __version__
            response = json.dumps(response).encode('utf-8')
            self.send(response)

        def reload(self):
            response = b'reload'
            self.send(response)

        def list_classifier(self):
            response = dict()
            response["status"] = "OK"
            response["result"] = {classifier: Server.classifiers[classifier]['enabled']
                                  for classifier in Server.classifiers}
            response = json.dumps(response).encode('utf-8')
            self.send(response)

        def set_classifier(self, classifier, value):
            response = dict()
            response["status"] = "OK"
            if value.lower() == "true":
                Server.classifiers[classifier]['enabled'] = True
            elif value.lower() == "false":
                Server.classifiers[classifier]['enabled'] = False
            else:
                response["status"] = "ERROR"
            response["result"] = [{classifier: Server.classifiers[classifier]['enabled']}]
            response = json.dumps(response).encode('utf-8')
            self.send(response)

        def md5_file(self, file_name=None):
            # This function is just for testing purpose
            hash_md5 = hashlib.md5(open(file_name, 'rb').read())
            response = dict()
            if hash_md5:
                response["status"] = "OK"
                response["result"] = hash_md5.hexdigest()
                response = json.dumps(response).encode('utf-8')
            else:
                response["status"] = "Error"
                response["result"] = ""
                response = json.dumps(response).encode('utf-8')
            self.send(response)

        def md5_stream(self):
            # This function is just for testing purpose
            hash_md5 = hashlib.md5()
            while True:
                data = self.receive()
                if data is None:
                    break
                hash_md5.update(data)
            response = dict()
            if hash_md5:
                response["status"] = "OK"
                response["result"] = hash_md5.hexdigest()
                response = json.dumps(response).encode('utf-8')
            else:
                response["status"] = "Error"
                response["result"] = ""
                response = json.dumps(response).encode('utf-8')
            self.send(response)

        def predict_stream(self, mid=None):
            stream = b''
            while True:
                data = self.receive()
                if data is None:
                    break
                else:
                    self.logger.debug("Data: {}...".format(data[:128]))
                stream += data
            stream = stream.decode('utf-8')
            multi_line = stream.split('\n')
            response = dict()
            response["status"] = "OK"
            response["mid"] = mid
            result = dict()
            for classifier_name in Server.classifiers.keys():
                if Server.classifiers[classifier_name]['enabled']:
                    result[classifier_name] = \
                        Server.classifiers[classifier_name]['class'].predict(multi_line)
            response["result"] = result
            self.logger.info("Result: {}".format(result))
            response = json.dumps(response).encode('utf-8')
            self.send(response)

        def predict_file(self,  mid=None, file_name=None):
            data = open(file_name, 'rb').read().decode('utf-8')
            multi_line = data.split('\n')
            response = dict()
            response["status"] = "OK"
            response["mid"] = mid
            result = dict()
            for classifier_name in Server.classifiers.keys():
                if Server.classifiers[classifier_name]['enabled']:
                    result[classifier_name] = \
                        Server.classifiers[classifier_name]['class'].predict(multi_line)
            response["result"] = result
            self.logger.info("Result: {}".format(result))
            response = json.dumps(response).encode('utf-8')
            self.send(response)

        def unknown_command(self):
            response = dict()
            response["status"] = "ERROR"
            response["result"] = "Unknown Command"
            response = json.dumps(response).encode('utf-8')
            self.send(response)

    class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
        allow_reuse_address = True
        pass