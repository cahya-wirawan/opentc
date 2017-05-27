import datetime
import logging
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn

from . import cnn_text_util


class CnnText(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    This code is based on Denny Britz's CNN implementation for text classification in Tensorflow:
    http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        self.logger = logging.getLogger(__name__)

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


class CnnTextEvaluator(object):
    """
     CnnTextEvaluator
     """
    def __init__(self, cfg, current_category):
        self.logger = logging.getLogger(__name__)
        self.cfg = cfg
        self.current_category = current_category
        self.x_raw = None
        self.x_test = None
        # Map data into vocabulary
        self.vocab_path = os.path.join(self.cfg['pre_trained_dir'][self.current_category], "vocab")
        self.vocab_processor = learn.preprocessing.VocabularyProcessor.restore(self.vocab_path)

        self.sess = None
        self.predictions = None
        self.dropout_keep_prob = None
        self.input_x = None
        checkpoint_file = tf.train.latest_checkpoint(self.cfg['pre_trained_dir'][self.current_category] + "/checkpoints")
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(
                # device_count={'GPU':0},
                allow_soft_placement=self.cfg['allow_soft_placement'],
                log_device_placement=self.cfg['log_device_placement'])
            self.sess = tf.Session(config=session_conf)
            with self.sess.as_default():
                # Load the saved meta graph and restore variables
                self.logger.debug("Load the checkpoint: {}".format(checkpoint_file))
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(self.sess, checkpoint_file)

                # Get the placeholders from the graph by name
                self.input_x = graph.get_operation_by_name("input_x").outputs[0]
                # input_y = graph.get_operation_by_name("input_y").outputs[0]
                self.dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

                # Tensors we want to evaluate
                self.predictions = graph.get_operation_by_name("output/predictions").outputs[0]

                # Tensors we want to evaluate
                self.scores = graph.get_operation_by_name("output/scores").outputs[0]

    def predict(self, x_raw=None):
        start = time.time()
        self.x_raw = x_raw
        self.x_test = np.array(list(self.vocab_processor.transform(self.x_raw)))

        # Generate batches for one epoch
        batches = cnn_text_util.batch_iter(list(self.x_test), self.cfg['batch_size'], 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []
        all_probabilities = None

        for x_test_batch in batches:
            batch_predictions_scores = self.sess.run([self.predictions, self.scores],
                                                {self.input_x: x_test_batch, self.dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions_scores[0]])
            probabilities = self.softmax(batch_predictions_scores[1])
            if all_probabilities is not None:
                all_probabilities = np.concatenate([all_probabilities, probabilities])
            else:
                all_probabilities = probabilities
        response = dict()
        response["predictions"] = [int(i) for i in all_predictions]
        response["probabilities"] = [all_probabilities[i][int(all_predictions[i])] for i in range(len(all_probabilities))]
        end = time.time()
        self.logger.info("Predict time: {} seconds".format(end - start))
        self.logger.debug("Response: {}".format(response))
        return response

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        if x.ndim == 1:
            x = x.reshape((1, -1))
        max_x = np.max(x, axis=1).reshape((-1, 1))
        exp_x = np.exp(x - max_x)
        return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))


class CnnTextTraining(object):
    """
     CnnTextTraining
     """
    def __init__(self, cfg):
        self.logger = logging.getLogger(__name__)
        self.logger.debug("CnnTextTraining init")
        self.cfg = cfg
        if self.cfg['word_embeddings']['default'] is not None:
            self.embedding_name = cfg['word_embeddings']['default']
            self.embedding_dimension = self.cfg['word_embeddings'][self.embedding_name]['dimension']
        else:
            self.embedding_dimension = self.cfg['word_embeddings']['embedding_dim']

    def fit(self, dataset, output):
        self.logger.debug("CnnTextTraining fit")

        x_text, y = dataset.load_data_labels()

        # Build vocabulary
        max_document_length = max([len(x.split(" ")) for x in x_text])
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
        x = np.array(list(vocab_processor.fit_transform(x_text)))

        # Randomly shuffle data
        np.random.seed(10)
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x_shuffled = x[shuffle_indices]
        y_shuffled = y[shuffle_indices]

        # Split train/test set
        # TODO: This is very crude, should use cross-validation
        dev_sample_index = -1 * int(self.cfg['dev_sample_percentage'] * float(len(y)))
        x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
        y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
        self.logger.debug("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
        self.logger.debug("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
                # device_count={'GPU':0},
                allow_soft_placement=self.cfg['allow_soft_placement'],
                log_device_placement=self.cfg['log_device_placement'])
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                cnn = CnnText(
                    sequence_length=x_train.shape[1],
                    num_classes=y_train.shape[1],
                    vocab_size=len(vocab_processor.vocabulary_),
                    embedding_size=self.embedding_dimension,
                    filter_sizes=list(map(int, self.cfg['filter_sizes'].split(","))),
                    num_filters=self.cfg['num_filters'],
                    l2_reg_lambda=self.cfg['l2_reg_lambda'])

                # Define Training procedure
                global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdamOptimizer(1e-3)
                grads_and_vars = optimizer.compute_gradients(cnn.loss)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

                # Keep track of gradient values and sparsity (optional)
                grad_summaries = []
                for g, v in grads_and_vars:
                    if g is not None:
                        grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name.replace(":","_")), g)
                        sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name.replace(":","_")), tf.nn.zero_fraction(g))
                        grad_summaries.append(grad_hist_summary)
                        grad_summaries.append(sparsity_summary)
                grad_summaries_merged = tf.summary.merge(grad_summaries)

                # Output directory for models and summaries
                timestamp = str(int(time.time()))
                out_dir = os.path.abspath(os.path.join(os.path.curdir, output))
                self.logger.debug("Writing to {}\n".format(out_dir))

                # Summaries for loss and accuracy
                loss_summary = tf.summary.scalar("loss", cnn.loss)
                acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

                # Train Summaries
                train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
                train_summary_dir = os.path.join(out_dir, "summaries", "train")
                train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

                # Dev summaries
                dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
                dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
                dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

                # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
                checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.cfg['num_checkpoints'])

                # Write vocabulary
                vocab_processor.save(os.path.join(out_dir, "vocab"))

                # Initialize all variables
                sess.run(tf.global_variables_initializer())
                if self.cfg['word_embeddings']['default'] is not None:
                    vocabulary = vocab_processor.vocabulary_
                    initW = None
                    if self.embedding_name == 'word2vec':
                        # load embedding vectors from the word2vec
                        self.logger.debug("Load word2vec file {}".format(self.cfg['word_embeddings']['word2vec']['path']))
                        initW = cnn_text_util.load_embedding_vectors_word2vec(vocabulary,
                                                                              self.cfg['word_embeddings']['word2vec']['path'],
                                                                              self.cfg['word_embeddings']['word2vec']['binary'])
                        self.logger.debug("word2vec file has been loaded")
                    elif self.embedding_name == 'glove':
                        # load embedding vectors from the glove
                        self.logger.debug("Load glove file {}".format(self.cfg['word_embeddings']['glove']['path']))
                        initW = cnn_text_util.load_embedding_vectors_glove(vocabulary,
                                                                           self.cfg['word_embeddings']['glove']['path'],
                                                                           self.embedding_dimension)
                        self.logger.debug("glove file has been loaded\n")
                    sess.run(cnn.W.assign(initW))

                def train_step(x_batch, y_batch):
                    """
                    A single training step
                    """
                    feed_dict = {
                        cnn.input_x: x_batch,
                        cnn.input_y: y_batch,
                        cnn.dropout_keep_prob: self.cfg['dropout_keep_prob']
                    }
                    _, step, summaries, loss, accuracy = sess.run(
                        [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                        feed_dict)
                    time_str = datetime.datetime.now().isoformat()
                    self.logger.debug("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                    train_summary_writer.add_summary(summaries, step)

                def dev_step(x_batch, y_batch, writer=None):
                    """
                    Evaluates model on a dev set
                    """
                    feed_dict = {
                        cnn.input_x: x_batch,
                        cnn.input_y: y_batch,
                        cnn.dropout_keep_prob: 1.0
                    }
                    step, summaries, loss, accuracy = sess.run(
                        [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                        feed_dict)
                    time_str = datetime.datetime.now().isoformat()
                    self.logger.debug("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                    if writer:
                        writer.add_summary(summaries, step)

                # Generate batches
                batches = cnn_text_util.batch_iter(
                    list(zip(x_train, y_train)), self.cfg['batch_size'], self.cfg['num_epochs'])
                # Training loop. For each batch...
                for batch in batches:
                    x_batch, y_batch = zip(*batch)
                    train_step(x_batch, y_batch)
                    current_step = tf.train.global_step(sess, global_step)
                    if current_step % self.cfg['evaluate_every'] == 0:
                        self.logger.debug("\nEvaluation:")
                        dev_step(x_dev, y_dev, writer=dev_summary_writer)
                        self.logger.debug("")
                    if current_step % self.cfg['checkpoint_every'] == 0:
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        self.logger.debug("Saved model checkpoint to {}\n".format(path))
