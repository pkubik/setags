import re
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

import setags.data.utils as utils


def encode_text(text: str, encoding: dict, default=utils.UNKNOWN_WORD_CODE):
    clean_text = re.sub(r'\W+', ' ', text).strip()
    tokens = clean_text.split()
    return [encoding.get(token, default) for token in tokens] + [0]


def prepare_data(filenames: list, data_dir: Path, train_fraction=1.0):
    input_dir = data_dir / utils.RAW_DATA_SUBDIR
    train_dir = data_dir / utils.TRAIN_DATA_SUBDIR
    test_dir = data_dir / utils.TEST_DATA_SUBDIR

    vocab = utils.load_vocabulary(data_dir)
    word_encoding = {value: i for i, value in enumerate(vocab)}

    for filename in filenames:
        name, _ = filename.split('.')
        with (input_dir / filename).open() as f:
            df = pd.read_csv(f)
        df['id'] = df['id'].apply(lambda x: name + str(x))

        if train_fraction == 1.0:
            train_df = df
        else:
            np.random.seed(0)
            train_df = df.sample(frac=train_fraction)

        store_tfrecords_from_df(name, word_encoding, train_df, train_dir)
        if train_fraction < 1.0:
            test_df = df.drop(train_df.index)
            store_tfrecords_from_df(name, word_encoding, test_df, test_dir)


def encoded_string_features_dict(name, value, word_encoding):
    tokens = encode_text(value, word_encoding)
    return {
        name: tf.train.Feature(int64_list=tf.train.Int64List(value=tokens)),
        name + '_length': tf.train.Feature(int64_list=tf.train.Int64List(value=[len(tokens)]))
    }


def string_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


def store_tfrecords_from_df(name: str, word_encoding: dict, train_df: pd.DataFrame, train_dir: Path):
    record_writer = tf.python_io.TFRecordWriter(str(train_dir / '{}.tfrecords'.format(name)))
    for _, row in train_df.iterrows():
        features = {
            'id': string_feature(row['id']),
            'original_title': string_feature(row['title']),
        }
        features.update(encoded_string_features_dict('title', row['title'], word_encoding))
        features.update(encoded_string_features_dict('content', row['content'], word_encoding))
        features.update(encoded_string_features_dict('tags', row['tags'], word_encoding))
        example = tf.train.Example(features=tf.train.Features(feature=features))
        record_writer.write(example.SerializeToString())
