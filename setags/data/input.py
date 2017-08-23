from pathlib import Path

import numpy as np
import tensorflow as tf
import setags.data.utils as utils
import logging

EMBEDDING_SIZE = 300

logger = logging.getLogger(__name__)


def create_input_fn(data_dir: Path, batch_size: int, for_train=True, num_epochs=1):
    if for_train:
        train_data_dir = data_dir / utils.TRAIN_DATA_SUBDIR
        filenames = [str(filename) for filename in train_data_dir.iterdir()]
    else:
        test_data_dir = data_dir / utils.TEST_DATA_SUBDIR
        filenames = [str(filename) for filename in test_data_dir.iterdir()]

    vocabulary = utils.load_vocabulary(data_dir)

    def embeddings_init_fn():
        logger.warning("Initializing words embeddings")
        return utils.load_embeddings_matrix(data_dir).astype(np.float32)

    def input_fn():
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs)

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        seq_fields = tf.parse_single_example(
            serialized_example,
            features={
                'id': tf.FixedLenFeature([], dtype=tf.string),
                'original_title': tf.FixedLenFeature([], dtype=tf.string),
                'title': tf.FixedLenSequenceFeature([], dtype=tf.int64, allow_missing=True),
                'title_length': tf.FixedLenFeature([], dtype=tf.int64),
                'content': tf.FixedLenSequenceFeature([], dtype=tf.int64, allow_missing=True),
                'content_length': tf.FixedLenFeature([], dtype=tf.int64),
                'tags': tf.FixedLenSequenceFeature([], dtype=tf.int64, allow_missing=True),
                'tags_length': tf.FixedLenFeature([], dtype=tf.int64),
            })

        # Shuffling disabled because dynamic_pad=True is not available
        # batched_fields = tf.train.shuffle_batch(
        #     fields, batch_size=batch_size, num_threads=2,
        #     capacity=utils.MIN_AFTER_DEQUEUE + 3 * batch_size,
        #     min_after_dequeue=utils.MIN_AFTER_DEQUEUE)
        batched_fields = tf.train.batch(
            seq_fields, batch_size=batch_size, num_threads=2,
            capacity=3 * batch_size, dynamic_pad=True)

        features = {key: batched_fields[key]
                    for key in ['id', 'original_title', 'title', 'title_length', 'content', 'content_length']}
        labels = {key: batched_fields[key] for key in ['tags', 'tags_length']}

        [embeddings_init] = tf.py_func(embeddings_init_fn, [], [tf.float32], stateful=True)
        embeddings_init.set_shape([len(vocabulary), EMBEDDING_SIZE])
        features['embeddings_initializer'] = embeddings_init

        return features, labels

    return input_fn
