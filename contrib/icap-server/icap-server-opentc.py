#!/bin/env python
# -*- coding: utf8 -*-

import os
import re
import magic
import socketserver
import multipart
import logging.config
import json
import argparse
import yaml
from opentc.client import Client
from multipart.multipart import parse_options_header
from pyicap import ICAPServer, BaseICAPRequestHandler


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


class ThreadingSimpleServer(socketserver.ThreadingMixIn, ICAPServer):
    pass


class ICAPHandler(BaseICAPRequestHandler):
    logger = logging.getLogger(__name__)
    remove_newline = re.compile(b'\r?\n')

    def opentc_OPTIONS(self):
        response = self.server.opentc.command("PING\n")
        response = json.loads(response.decode('utf-8'))
        if response["status"] == "OK":
            self.logger.debug("OPTIONS Ping response: {}".format(response))
        else:
            self.logger.debug("OPTIONS Ping response: the OpenTC server is not responding")
        self.set_icap_response(200)
        self.set_icap_header(b'Methods', b'REQMOD')
        self.set_icap_header(b'Service', b'PyICAP Server 1.0')
        self.send_headers(False)

    def opentc_REQMOD(self):
        self.multipart_data = None
        self.last_form_field = None
        self.big_chunk = b''
        try:
            response = self.server.opentc.command("PING\n")
            response = json.loads(response.decode('utf-8'))
            self.logger.debug("REQMOD Ping response: {}".format(response))
        except OSError as err:
            self.logger.error("OS error: {0}".format(err))

        def on_part_begin():
            self.multipart_data = dict()
            self.multipart_data[b'Content'] = b''
            self.logger.debug("on_part_begin")

        def on_part_data(data, start, end):
            self.multipart_data[b'Content'] += data[start:end]
            self.logger.debug("on_part_data")

        def on_part_end():
            self.logger.debug("on_part_end")
            for key in self.multipart_data.keys():
                if key == b'Content':
                    mime_type = magic.from_buffer(self.multipart_data[b'Content'], mime=True)
                    self.logger.debug("Content mime_type: {}".format(mime_type))
                    if b'Content-Type' in self.multipart_data:
                        content_type = [ct.strip() for ct in self.multipart_data[b'Content-Type'].split(b';')]
                        if b'text/plain' in content_type:
                            content = self.remove_newline.sub(b' ', self.multipart_data[b'Content'])
                            response = self.server.opentc.predict_stream(content)
                            response = json.loads(response.decode('utf-8'))
                            self.logger.debug("on_part_end predict_stream response: {}".format(response))
                else:
                    self.logger.debug("{}: {}".format(key, self.multipart_data[key]))
            return "end"

        def on_header_field(data, start, end):
            self.last_form_field = data[start:end]
            self.logger.debug("on_header_field")

        def on_header_value(data, start, end):
            self.multipart_data[self.last_form_field] = data[start:end]
            self.logger.debug("on_header_value")

        self.set_icap_response(200)

        # self.set_enc_request(b' '.join(self.enc_req))
        for h in self.enc_req_headers:
            for v in self.enc_req_headers[h]:
                self.set_enc_header(h, v)

        # Copy the request body (in case of a POST for example)
        if not self.has_body:
            self.set_enc_request(b' '.join(self.enc_req))
            self.send_headers(False)
            return
        if self.preview:
            prevbuf = b''
            while True:
                chunk = self.read_chunk()
                if chunk == b'':
                    break
                prevbuf += chunk
            if self.ieof:
                self.send_headers(True)
                if len(prevbuf) > 0:
                    self.write_chunk(prevbuf)
                self.write_chunk(b'')
                return
            self.cont()
            self.set_enc_request(b' '.join(self.enc_req))
            self.send_headers(True)
            if len(prevbuf) > 0:
                self.write_chunk(prevbuf)
            while True:
                chunk = self.read_chunk()
                self.write_chunk(chunk)
                if chunk == b'':
                    break
        else:
            # Parse the Content-Type header to get the multipart boundary.
            content_type, params = parse_options_header(self.enc_req_headers[b'content-type'][0])
            boundary = params.get(b'boundary')

            # Callbacks dictionary.
            callbacks = {
                'on_part_begin': on_part_begin,
                'on_part_data': on_part_data,
                'on_part_end': on_part_end,
                'on_header_field': on_header_field,
                'on_header_value': on_header_value
            }
            parser = multipart.MultipartParser(boundary, callbacks)
            while True:
                chunk = self.read_chunk()
                if chunk == b'':
                    break
                self.big_chunk += chunk

            size = len(self.big_chunk)
            start = 0
            while size > 0:
                end = min(size, 1024 * 1024)
                parser.write(self.big_chunk[start:end])
                size -= end
                start = end

            self.set_enc_request(b' '.join(self.enc_req))
            self.send_headers(True)
            self.write_chunk(self.big_chunk)


if __name__ == '__main__':
    config_directories = [os.curdir, os.path.expanduser("~/.icap-server-opentc"), "/etc/icap-server-opentc",
                          os.environ.get("ICAPSERVER_CONF_DIR")]
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--icap_server_address", help="define the address of the icap server")
    parser.add_argument("-A", "--opentc_server_address", help="define the address of the opentc server")
    parser.add_argument("-C", "--configuration_file", help="set the configuration file")
    parser.add_argument("-l", "--log_configuration_file", help="set the log configuration file")
    parser.add_argument("-p", "--icap_server_port", help="define the port number which the icap server uses to listen")
    parser.add_argument("-P", "--opentc_server_port", help="define the port number which "
                                                           "the opentc server uses to listen")
    args = parser.parse_args()
    setup_logging(config_directories=config_directories)
    logger = logging.getLogger(__name__)
    cfg = setup_config(config_directories=config_directories, config_file=args.configuration_file)

    if args.icap_server_address:
        icap_server_address = args.icap_server_address
    else:
        icap_server_address = cfg["icap_server"]["address"]
    if args.icap_server_port:
        icap_server_port = args.icap_server_port
    else:
        icap_server_port = cfg["icap_server"]["port"]

    if args.opentc_server_address:
        opentc_server_address = args.opentc_server_address
    else:
        opentc_server_address = cfg["opentc_server"]["address"]
    if args.opentc_server_port:
        opentc_server_port = args.opentc_server_port
    else:
        opentc_server_port = cfg["opentc_server"]["port"]

    server = ThreadingSimpleServer((icap_server_address.encode('utf-8'), icap_server_port), ICAPHandler)
    server.opentc = Client(address=opentc_server_address, port=opentc_server_port)
    try:
        while 1:
            server.handle_request()
    except KeyboardInterrupt:
        logger.info("The icap server quits")
