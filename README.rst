=================================
Open Text Classification (OpenTC)
=================================
This is a text classification engine using several algorithms in machine learning. Following algorithms will be
supported:

- Naive Bayes (Scikit-learn)
- Support Vector Machine (Scikit-learn)
- Convolutional Neural Network (Tensorflow)
- FastText (Facebookresearch)

The engine is running as a server listening on command and text to be classified. By default it listens on localhost 
port 3333, but it can be changed in the yaml configuration file. 


Requirements
============
- Python 3.x
- numpy
- protobuf
- pyparsing
- PyYAML
- scikit-learn
- scipy
- tensorflow 1.x


opentcd
=======

synopsis
========

opentcd.py

Description
===========
The daemon listens for incoming connections on TCP socket and classify files or text string on demand. 
It reads the configuration from /etc/opentc/opentc.yml


Commands
--------
The command uses a newline character as the delimiter. If opentcd.py doesn't recognize the command,
or the command doesn't follow the requirements specified below, it will reply with an error message, but still wait 
for the next commands (this behaviour can be changed in the future).

PING
~~~~
Check the server's state. It should reply with "PONG".

VERSION
~~~~~~~
Print the program version

RELOAD
~~~~~~
Reload the engine

LIST_CLASSIFIER
~~~~~~~~~~~~~~~
List the supported classifiers (at the moment there are three classifiers
supported: Bayesian, Support Vector Machine and Convolutional Neural Network). It shows also 
the status of classifier, either True (enabled) or False (disabled).

SET_CLASSIFIER
~~~~~~~~~~~~~~
Enabled or disabled the specific classifier

PREDICT_STREAM
~~~~~~~~~~~~~~
Classify text streams. It uses a new line character as delimiter for every sentences. 

PREDICT_FILE
~~~~~~~~~~~~
Classify file. It uses a new line character as delimiter for every sentences

CLOSE
~~~~~
Close the connection

