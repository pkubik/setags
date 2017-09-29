import tensorflow as tf

from setags.data.input import EMBEDDING_SIZE
from setags.data.utils import BIOTag
from setags.utils import DictWrapper


class Features(DictWrapper):
    def __init__(self):
        self.id = None
        self.title = None
        self.title_length = None
        self.content = None
        self.content_length = None


class Labels(DictWrapper):
    def __init__(self):
        self.title_bio = None
        self.content_bio = None


class Params(DictWrapper):
    def __init__(self):
        self.num_epochs = 100
        self.batch_size = 64
        self.max_word_idx = None
        self.num_rnn_units = 300
        self.learning_rate = 0.002


BIO_ENCODING_SIZE = sum(1 for _ in BIOTag)


DEFAULT_PARAMS = Params().as_dict()


def model_fn(mode, features, labels, params):
    _params = Params.from_dict(params)
    _features = Features.from_dict(features)

    _labels = None
    if mode != tf.estimator.ModeKeys.PREDICT:
        _labels = Labels.from_dict(labels)

    return build_model(mode, _features, _labels, _params)


def build_model(mode: tf.estimator.ModeKeys,
                features: Features,
                labels: Labels,
                params: Params) -> tf.estimator.EstimatorSpec:

    with tf.device("/cpu:0"):
        embeddings = tf.placeholder(tf.float32, [None, EMBEDDING_SIZE], name='embeddings')

    embedded_title = tf.nn.embedding_lookup(embeddings, tf.nn.relu(features.title))
    embedded_content = tf.nn.embedding_lookup(embeddings, tf.nn.relu(features.content))

    with tf.variable_scope("encoder"):
        with tf.variable_scope("title"):
            title_encoder = RNNLayer(embedded_title, features.title_length, params.num_rnn_units)
        with tf.variable_scope("content"):
            content_encoder = RNNLayer(
                embedded_content, features.content_length, params.num_rnn_units, title_encoder.final_states_tuple)

    with tf.variable_scope("output"):
        title_bio_logits = tf.layers.dense(title_encoder.outputs, BIO_ENCODING_SIZE)
        content_bio_logits = tf.layers.dense(content_encoder.outputs, BIO_ENCODING_SIZE)
        title_bio_predicitons = tf.argmax(title_bio_logits, -1)
        content_bio_predicitons = tf.argmax(content_bio_logits, -1)

    # Assign a default value to the train_op and loss to be passed for modes other than TRAIN
    loss = None
    train_op = None
    eval_metric_ops = None
    # Following part of the network will be constructed only for training
    if mode != tf.estimator.ModeKeys.PREDICT:
        hot_title_bio = tf.one_hot(labels.title_bio, BIO_ENCODING_SIZE)
        hot_content_bio = tf.one_hot(labels.content_bio, BIO_ENCODING_SIZE)
        title_mask = tf.sequence_mask(features.title_length, tf.reduce_max(features.title_length), tf.float32)
        content_mask = tf.sequence_mask(features.content_length, tf.reduce_max(features.content_length), tf.float32)
        title_bio_loss = tf.losses.softmax_cross_entropy(hot_title_bio, title_bio_logits, title_mask)
        content_bio_loss = tf.losses.softmax_cross_entropy(hot_content_bio, content_bio_logits, content_mask)

        loss = tf.losses.get_total_loss()
        tf.summary.scalar('title_loss', title_bio_loss)
        tf.summary.scalar('content_loss', content_bio_loss)

        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=params.learning_rate,
            optimizer="Adam")

        if mode == tf.estimator.ModeKeys.EVAL:
            title_accuracy = tf.metrics.accuracy(labels.title_bio, title_bio_predicitons, title_mask)
            content_accuracy = tf.metrics.accuracy(labels.content_bio, content_bio_predicitons, content_mask)

            eval_metric_ops = {
                'title_accuracy': title_accuracy,
                'content_accuracy': content_accuracy
            }

    predictions = {
        'id': features.id,
        'title': features.title,
        'title_length': features.title_length,
        'title_bio': title_bio_predicitons,
        'content': features.content,
        'content_length': features.content_length,
        'content_bio': content_bio_predicitons
    }

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)


class RNNLayer:
    def __init__(self, inputs: tf.Tensor, inputs_lengths: tf.Tensor, num_hidden: int, initial_states: tuple = None):
        fw_cell = tf.nn.rnn_cell.GRUCell(num_hidden, activation=tf.nn.tanh)
        bw_cell = tf.nn.rnn_cell.GRUCell(num_hidden, activation=tf.nn.tanh)
        if initial_states is not None:
            fw_initial_state, bw_initial_state = initial_states
        else:
            fw_initial_state, bw_initial_state = None, None

        self.outputs_tuple, self.final_states_tuple = tf.nn.bidirectional_dynamic_rnn(
            fw_cell, bw_cell, inputs, inputs_lengths,
            initial_state_fw=fw_initial_state,
            initial_state_bw=bw_initial_state,
            dtype=tf.float32)

    @property
    def outputs(self):
        return tf.reduce_sum(self.outputs_tuple, axis=0)
