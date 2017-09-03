from pathlib import Path

import numpy as np
import tensorflow as tf
import setags.data.utils as utils
import logging

EMBEDDING_SIZE = 300
FILENAME_INPUT_PRODUCER_SEED = 1

logger = logging.getLogger(__name__)


def create_input_fn(data_subdir: Path, data_dir: Path, batch_size: int, for_train=True, num_epochs=1):
    filenames = [str(filename) for filename in data_subdir.iterdir()]

    vocabulary = utils.load_vocabulary(data_dir)

    def embeddings_init_fn():
        logger.warning("Initializing words embeddings")
        return utils.load_embeddings_matrix(data_dir).astype(np.float32)

    def input_fn():
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs,
                                                        seed=FILENAME_INPUT_PRODUCER_SEED)

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        if for_train:
            examples_batch = tf.train.shuffle_batch(
                [serialized_example], batch_size=batch_size, num_threads=3,
                capacity=utils.MIN_AFTER_DEQUEUE + 5 * batch_size,
                min_after_dequeue=utils.MIN_AFTER_DEQUEUE)
        else:
            examples_batch = tf.train.batch(
                [serialized_example], batch_size=batch_size, num_threads=3, capacity=5 * batch_size)

        example_fields = tf.parse_example(
            examples_batch,
            features={
                'id': tf.FixedLenFeature([], dtype=tf.string),
                'title': tf.FixedLenSequenceFeature([], dtype=tf.int64, allow_missing=True),
                'title_length': tf.FixedLenFeature([], dtype=tf.int64),
                'content': tf.FixedLenSequenceFeature([], dtype=tf.int64, allow_missing=True),
                'content_length': tf.FixedLenFeature([], dtype=tf.int64),
                'tags': tf.FixedLenSequenceFeature([], dtype=tf.int64, allow_missing=True),
                'tags_length': tf.FixedLenFeature([], dtype=tf.int64),
            })

        features = {key: example_fields[key]
                    for key in ['id', 'title', 'title_length', 'content', 'content_length']}
        labels = {key: example_fields[key] for key in ['tags', 'tags_length']}

        [embeddings_init] = tf.py_func(embeddings_init_fn, [], [tf.float32], stateful=True)
        embeddings_init.set_shape([len(vocabulary), EMBEDDING_SIZE])
        features['embeddings_initializer'] = embeddings_init

        return features, labels

    return input_fn
