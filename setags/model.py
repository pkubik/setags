import tensorflow as tf

from setags.utils import DictWrapper


class Features(DictWrapper):
    def __init__(self):
        self.id = None
        self.original_title = None
        self.title = None
        self.title_length = None
        self.content = None
        self.content_length = None


class Labels(DictWrapper):
    def __init__(self):
        self.tags = None
        self.tags_length = None


class Params(DictWrapper):
    def __init__(self):
        self.num_epochs = 20
        self.batch_size = 50
        self.num_rnn_units = 100
        self.learning_rate = 0.04
        self.dropout_rate = 0.4


DEFAULT_PARAMS = Params().as_dict()


def model_fn(mode, features, labels, params):
    p = Params.from_dict(params)
    assert isinstance(p, Params)

    f = Features.from_dict(features)
    assert isinstance(f, Features)

    l = Labels()
    if mode != tf.estimator.ModeKeys.PREDICT:
        l = Labels.from_dict(labels)
        assert isinstance(l, Labels)

    # Assign a default value to the train_op and loss to be passed for modes other than TRAIN
    loss = None
    train_op = None
    eval_metric_ops = None
    # Following part of the network will be constructed only for training
    if mode != tf.estimator.ModeKeys.PREDICT:
        loss = tf.constant(13.0)
        train_op = tf.Print(f.title, [f.original_title[0], f.title[0], tf.shape(f.title), f.title_length[0]])
        # train_op = tf.contrib.layers.optimize_loss(
        #     loss=loss,  # Total loss from the losses collection
        #     global_step=tf.contrib.framework.get_global_step(),
        #     learning_rate=p.learning_rate,
        #     optimizer="Adagrad")

        if mode == tf.estimator.ModeKeys.EVAL:
            eval_metric_ops = {
                'accuracy': tf.constant(44.0)
            }

    predictions = {}

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)
