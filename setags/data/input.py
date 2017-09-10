from collections import defaultdict
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import tensorflow as tf
import setags.data.utils as utils
import logging

EMBEDDING_SIZE = 300
FILENAME_INPUT_PRODUCER_SEED = 1

log = logging.getLogger(__name__)


def create_input_fn(data_subdir: Path, batch_size: int, for_train=True, num_epochs=1):
    filenames = [str(filename) for filename in data_subdir.iterdir()]

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

        return features, labels

    return input_fn


class PredictionInput:
    def __init__(self, data_file: Path, data_dir: Path, vocabulary: list, batch_size: int):
        self.data_dir = data_dir
        self.original_vocab_size = len(vocabulary)
        self.word_encoding = defaultdict(lambda: len(self.word_encoding))
        self.word_encoding.update({value: i for i, value in enumerate(vocabulary)})

        self._create_input_fn(batch_size, data_file)
        self._create_hooks()

    def _create_input_fn(self, batch_size, data_file):
        with data_file.open() as f:
            df = pd.read_csv(f)

        ids = []
        titles = []
        contents = []
        max_title_length = 0
        max_content_length = 0
        for _, row in df.iterrows():
            ids.append(row.id)
            encoded_title = utils.encode_text(row.title, self.word_encoding)
            titles.append(encoded_title)
            encoded_content = utils.encode_text(row.content, self.word_encoding)
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
        self.input_fn = tf.estimator.inputs.numpy_input_fn(data, batch_size=batch_size, shuffle=False)

    def _create_hooks(self):
        embedding_matrix = utils.load_embeddings_matrix(self.data_dir)

        words = utils.encoding_as_list(self.word_encoding)[self.original_vocab_size:]
        if len(words) > 0:
            vocab_ext_file = tempfile.NamedTemporaryFile(mode='w+t', prefix='vocab-', delete=False)
            log.info("Storing new words in '{}'".format(vocab_ext_file.name))
            for word in words:
                vocab_ext_file.write(str(word) + '\n')
            vocab_ext_file.close()

            embedding_model = utils.load_embedding_model(self.data_dir)
            new_vectors = utils.create_embedding_vectors(words, embedding_model)
            new_matrix = np.array(new_vectors)

            all_embeddings = np.concatenate((embedding_matrix, new_matrix))
        else:
            all_embeddings = embedding_matrix

        self.hooks = [create_embedding_feed_hook(all_embeddings)]


def encode_as_array(sequences: list, max_sequence_length):
    ret = np.zeros([len(sequences), max_sequence_length], dtype=np.int64)
    for r, seq in enumerate(sequences):
        for c, elem in enumerate(seq):
            ret[r][c] = elem

    return ret


def create_embedding_feed_hook(embedding_matrix):
    def feed_fn():
        return {
            'embeddings:0': embedding_matrix
        }
    return tf.train.FeedFnHook(feed_fn=feed_fn)
