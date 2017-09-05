import tensorflow as tf

from setags.data.input import EMBEDDING_SIZE
from setags.utils import DictWrapper


class Features(DictWrapper):
    def __init__(self):
        self.id = None
        self.title = None
        self.title_length = None
        self.content = None
        self.content_length = None
        self.embeddings_initializer = None


class Labels(DictWrapper):
    def __init__(self):
        self.tags = None
        self.tags_length = None


class Params(DictWrapper):
    def __init__(self):
        self.num_epochs = 100
        self.batch_size = 64
        self.max_word_idx = None
        self.max_tag_idx = None
        self.num_rnn_units = 500
        self.learning_rate = 0.002


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
        if features.embeddings_initializer is not None:
            embeddings = tf.get_variable(
                'embeddings', initializer=features.embeddings_initializer, trainable=False)
        else:
            embeddings = tf.get_variable(
                'embeddings', shape=[params.max_word_idx, EMBEDDING_SIZE], trainable=False)

    embedded_title = tf.nn.embedding_lookup(embeddings, tf.nn.relu(features.title))
    embedded_content = tf.nn.embedding_lookup(embeddings, tf.nn.relu(features.content))

    with tf.variable_scope("encoder"):
        with tf.variable_scope("title"):
            title_encoder = RNNLayer(embedded_title, features.title_length, params.num_rnn_units)
        with tf.variable_scope("content"):
            content_encoder = RNNLayer(
                embedded_content, features.content_length, params.num_rnn_units, title_encoder.final_state)

    with tf.variable_scope("decoder"):
        first_tag_logits = tf.layers.dense(content_encoder.final_state, params.max_tag_idx)
        first_tag_prediction = tf.argmax(first_tag_logits, -1)

    # Assign a default value to the train_op and loss to be passed for modes other than TRAIN
    loss = None
    train_op = None
    eval_metric_ops = None
    # Following part of the network will be constructed only for training
    if mode != tf.estimator.ModeKeys.PREDICT:
        first_tag = tf.nn.relu(labels.tags[:, 0])
        first_one_hot_tag = tf.one_hot(first_tag, params.max_tag_idx, dtype=tf.float32)
        tf.losses.softmax_cross_entropy(first_one_hot_tag, first_tag_logits)

        loss = tf.losses.get_total_loss()
        tf.summary.scalar('loss', loss)

        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=params.learning_rate,
            optimizer="Adam")

        if mode == tf.estimator.ModeKeys.EVAL:
            accuracy = tf.metrics.accuracy(first_tag, first_tag_prediction)

            eval_metric_ops = {
                'accuracy': accuracy
            }

    predictions = {
        'id': features.id,
        'tags': first_tag_prediction,
    }

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)


class RNNLayer:
    def __init__(self, inputs: tf.Tensor, inputs_lengths: tf.Tensor, num_hidden: int, initial_state: tf.Tensor = None):
        cell = tf.nn.rnn_cell.GRUCell(num_hidden, activation=tf.nn.tanh)
        self.outputs, self.final_state = tf.nn.dynamic_rnn(
            cell, inputs, inputs_lengths, initial_state, dtype=tf.float32)
