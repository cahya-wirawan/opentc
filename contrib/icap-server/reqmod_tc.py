#!/bin/env python
# -*- coding: utf8 -*-

import os
import re
import magic
import socketserver
import multipart
import logging
import json
import argparse
from opentc.client import Client
from opentc import setup_logging
from multipart.multipart import parse_options_header
from pyicap import ICAPServer, BaseICAPRequestHandler


class ThreadingSimpleServer(socketserver.ThreadingMixIn, ICAPServer):
    pass


class ICAPHandler(BaseICAPRequestHandler):
    address, port = "localhost", 3333
    logger = logging.getLogger(__name__)
    tcc = Client(address=address, port=port)
    remove_newline = re.compile(b'\r?\n')

    def example_OPTIONS(self):
        self.set_icap_response(200)
        self.set_icap_header(b'Methods', b'REQMOD')
        self.set_icap_header(b'Service', b'PyICAP Server 1.0')
        self.send_headers(False)

    def example_REQMOD(self):
        self.multipart_data = None
        self.last_form_field = None
        self.big_chunk = b''
        response = self.tcc.command("PING\n")
        response = json.loads(response.decode('utf-8'))
        self.logger.debug("REQMOD Ping response: {}".format(response))

        def on_part_begin():
            self.multipart_data = dict()
            self.multipart_data[b'Content'] = b''
            logger.debug("on_part_begin")

        def on_part_data(data, start, end):
            self.multipart_data[b'Content'] += data[start:end]
            logger.debug("on_part_data")

        def on_part_end():
            logger.debug("on_part_end")
            for key in self.multipart_data.keys():
                if key == b'Content':
                    mime_type = magic.from_buffer(self.multipart_data[b'Content'], mime=True)
                    logger.debug("Content mime_type: {}".format(mime_type))
                    if b'Content-Type' in self.multipart_data:
                        content_type = [ct.strip() for ct in self.multipart_data[b'Content-Type'].split(b';')]
                        if b'text/plain' in content_type:
                            content = self.remove_newline.sub(b' ', self.multipart_data[b'Content'])
                            response = self.tcc.predict_stream(content)
                            response = json.loads(response.decode('utf-8'))
                            logger.debug("on_part_end predict_stream response: {}".format(response))
                else:
                    logger.debug("{}: {}".format(key, self.multipart_data[key]))
            return "end"

        def on_header_field(data, start, end):
            self.last_form_field = data[start:end]
            logger.debug("on_header_field")

        def on_header_value(data, start, end):
            self.multipart_data[self.last_form_field] = data[start:end]
            logger.debug("on_header_value")

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
    config_directories = [os.curdir, os.path.expanduser("~/.opentc"), "/etc/opentc", os.environ.get("OPENTC_CONF_DIR")]
    setup_logging(config_directories=config_directories)
    logger = logging.getLogger(__name__)

    port = 13440

    server = ThreadingSimpleServer((b'', port), ICAPHandler)
    try:
        while 1:
            server.handle_request()
    except KeyboardInterrupt:
        logger.debug("Finished")
