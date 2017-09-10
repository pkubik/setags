from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import setags.data.utils as utils
import logging

EMBEDDING_SIZE = 300
FILENAME_INPUT_PRODUCER_SEED = 1

logger = logging.getLogger(__name__)


def create_input_fn(data_subdir: Path, data_dir: Path, vocab_size: int, batch_size: int, for_train=True, num_epochs=1):
    filenames = [str(filename) for filename in data_subdir.iterdir()]

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
                'tags_length': tf.FixedLenFeature([], dtype=tf.int64)
            })

        features = {key: example_fields[key]
                    for key in ['id', 'title', 'title_length', 'content', 'content_length']}
        labels = {key: example_fields[key] for key in ['tags', 'tags_length']}

        [embeddings_init] = tf.py_func(embeddings_init_fn, [], [tf.float32], stateful=True)
        embeddings_init.set_shape([vocab_size, EMBEDDING_SIZE])
        features['embeddings_initializer'] = embeddings_init

        return features, labels

    return input_fn


def create_input_fn_for_prediction(data_file: Path, data_dir: Path, batch_size: int):
    vocabulary = utils.load_vocabulary(data_dir)
    word_encoding = {value: i for i, value in enumerate(vocabulary)}

    with data_file.open() as f:
        df = pd.read_csv(f)

    ids = []
    titles = []
    contents = []
    max_title_length = 0
    max_content_length = 0
    for _, row in df.iterrows():
        ids.append(row.id)
        encoded_title = utils.encode_text(row.title, word_encoding)
        titles.append(encoded_title)
        encoded_content = utils.encode_text(row.content, word_encoding)
        contents.append(encoded_content)
        max_title_length = max(max_title_length, len(encoded_title))
        max_content_length = max(max_content_length, len(encoded_content))

    title_array = encode_as_array(titles, max_title_length)
    content_array = encode_as_array(contents, max_content_length)

    data = {
        'id': np.array(ids),
        'title': title_array,
        'content': content_array
    }

    return tf.estimator.inputs.numpy_input_fn(data, batch_size=batch_size, shuffle=False)


def encode_as_array(sequences: list, max_sequence_length):
    ret = np.zeros([len(sequences), max_sequence_length], dtype=np.int64)
    for r, seq in enumerate(sequences):
        for c, elem in enumerate(seq):
            ret[r][c] = elem

    return ret
