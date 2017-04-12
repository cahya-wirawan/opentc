
# icap-server-opentc.py
The Internet Content Adaptation Protocol (ICAP) is a lightweight HTTP-like protocol specified in RFC 3507 which 
is used to extend transparent proxy servers. It is used generally for a virus scanner or content filtering. 

The purpose of this icap-server-opentc is to demonstrate one of the usages of the OpenTC server.  In this case, the 
icap server is used as Data Leak Prevention (DLP). It is listening on the icap server's default port 1344.
The squid-cache is used as the http/s proxy (it can be also any other proxy servers). It is configured to connect 
to the icap server for the content filtering. Any out going http traffics through the squid proxy will be sent to 
the icap server, which will then forward it to the OpenTC server. OpenTC server analyses the data and classify it 
based on the pre-trained data. The result of the text classification is sent back to the icap server, which will 
decide if the outgoing traffic should be blocked or allowed.  

## Requirements
- Python 3.x
- opentc
- PyYAML
- pyicap 1.0b1
- python-magic
- python-multipart


## TODO
- the icap server should monitor the availability of the OpenTC server. In case it is not up or running, thís icap 
server should try to reconnect it again several times in difference interval (i.e: the interval of the first 3 
reconnection could be 10 seconds, and after 3 unsuccessful attempts to reconnect, the interval connection time 
should be changed to 300 seconds). Currently, the icap server has to be restarted manually after the OpenTC server
is died or restarted.
- Currently the decision to take, either the traffic is blocked or allowed, is implemented using manual 
"for_loop-if-elif" sequences, which maybe not easy to understand or error prone. In the future, the software would
use a rule engine (if there is any) to simplify the creation of complex rules.