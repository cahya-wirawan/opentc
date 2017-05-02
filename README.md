# Open Text Classification (OpenTC)

OpenTC is a text classification engine using machine learning. It is designed as client-server architecture and uses 
python libraries scikit-learn and tensorflow for it's machine learning algorithms. 
Currently following algorithms are supported:

- Naive Bayes
- Support Vector Machine
- Convolutional Neural Network

In the future it will also support FastText from Facebookresearch. 

The engine is running as a server listening on command and text to be classified. By default it listens on localhost 
port 3333, but it can be changed in the yaml configuration file. 

OpenTC can be used for example for text classification (a demo website for this purpose is available online 
[OpenTC demo](http://opentc.oldjava.org/demo/)), or for other purposes such as Data Leak Prevention (DLP). 
An example of implementation for the DLP has been created as ICAP Server: 
[opentc-icap](https://github.com/cahya-wirawan/opentc-icap) 


## Requirements
- Python 3.x
- numpy
- pyparsing
- PyYAML
- scikit-learn
- scipy
- tensorflow 1.x

## How to use

### Installation
Install the module using pip:

    $ pip install opentc
    
or clone the repository
    
    $ git clone https://github.com/cahya-wirawan/opentc.git
    $ cd opentc
    $ python setup.py install


### opentc

#### synopsis
opentc

#### Description
The command line to train the application based on the datasets define in the configuration file. The result
of the training (pre-trained data) can be used for the opentcd server.

#### Usage

    $ python opentc -h
    usage: opentc [-h] [-c CLASSIFIER] [-C CONFIGURATION_FILE] [-d DATASET]
                  [-l LOG_CONFIGURATION_FILE]
    
    optional arguments:
      -h, --help            show this help message and exit
      -c CLASSIFIER, --classifier CLASSIFIER
                            set classifier to use for the training (support
                            currently bayesian, svm or cnn)
      -C CONFIGURATION_FILE, --configuration_file CONFIGURATION_FILE
                            set the configuration file
      -d DATASET, --dataset DATASET
                            set dataset to use for the training
      -l LOG_CONFIGURATION_FILE, --log_configuration_file LOG_CONFIGURATION_FILE
                            set the log configuration file


### opentcd

#### synopsis
opentcd

#### Description
The daemon listens for incoming connections on TCP port (default is 3333) and classify files or text string on 
demand. It reads a configuration file in the following order: ./opentc.yml, ~/.opentc/opentc.yml or 
/etc/opentc/opentc.yml.

#### Usage
Opentcd uses the configuration file opentc.yml to define allmost all possible configuration. Only few setup
can be overridden in command line options.

List of arguments:

    $ python opentcd -h
    usage: opentcd [-h] [-a ADDRESS] [-C CONFIGURATION_FILE]
                   [-l LOG_CONFIGURATION_FILE] [-p PORT] [-t TIMEOUT]
    
    optional arguments:
      -h, --help            show this help message and exit
      -a ADDRESS, --address ADDRESS
                            define the address for the server
      -C CONFIGURATION_FILE, --configuration_file CONFIGURATION_FILE
                            set the configuration file
      -l LOG_CONFIGURATION_FILE, --log_configuration_file LOG_CONFIGURATION_FILE
                            set the log configuration file
      -p PORT, --port PORT  define the port number which the server uses to listen
      -t TIMEOUT, --timeout TIMEOUT
                            define the time out

Run it as background application:
    
    $ python opentcd&
    2017-05-02 13:33:22,276 - opentc.core.classifier.cnn_text - DEBUG - Load the checkpoint: 
    data/input/cnn_twenty_newsgroup_20170301_090000-all/checkpoints/model-2210
    INFO:tensorflow:Restoring parameters from data/input/cnn_twenty_newsgroup_20170301_090000-all/checkpoints/model-2210
    2017-05-02 13:33:23,899 - tensorflow - INFO - Restoring parameters 
    from data/input/cnn_twenty_newsgroup_20170301_090000-all/checkpoints/model-2210
    2017-05-02 13:33:27,375 - __main__ - INFO - Server start
    2017-05-02 13:33:28,019 - opentc.core.server - INFO - Server loop running in thread: Thread-1




#### datasets and pre-trained data
The configuration file defines the path to the datasets and pre-trained data. A pre-trained data for testing
purpose can be downloaded from [data](https://NoFile.io/f/6ZkDvJH0nT4), it is around 1.4GB. Just uncompress it 
and change the path to the pre-trained data in opentc.yml file accordingly.

#### Commands
The command uses a newline character as the delimiter. If opentcd doesn't recognize the command, 
or the command doesn't follow the requirements specified below, it will reply with an error message, but still wait 
for the next commands (this behaviour can be changed in the future).

##### PING
Check the server's state. It should reply with "PONG".

##### VERSION
Print the program version

##### RELOAD
Reload the engine

##### LIST_CLASSIFIER
List the supported classifiers (at the moment there are three classifiers
supported: Bayesian, Support Vector Machine and Convolutional Neural Network). It shows also 
the status of classifier, either True (enabled) or False (disabled).

##### SET_CLASSIFIER
Enabled or disabled the specific classifier

##### PREDICT_STREAM
Classify text streams. It uses a new line character as delimiter for every sentences. 

##### PREDICT_FILE
Classify file. It uses a new line character as delimiter for every sentences

##### CLOSE
Close the connection

## Todo
- Multilabel classification
- Include FastText from Facebookresearch
- 