from pathlib import Path
from typing import Iterable

import tensorflow as tf

import setags.data.input as di
import setags.data.utils as du
from setags.model import model_fn, DEFAULT_PARAMS


class Estimator:
    def __init__(self, model_dir: Path, data_dir: Path, param_overrides: dict = None):
        if param_overrides is None:
            param_overrides = {}

        self.model_dir = model_dir

        self.params = {}
        self.params.update(DEFAULT_PARAMS)
        self.params.update(du.load_params(model_dir))
        self.params.update(param_overrides)

        self.num_epochs = self.params['num_epochs']
        self.batch_size = self.params['batch_size']

        self.data_dir = data_dir

        self.vocabulary = du.load_list(self.data_dir / du.VOCAB_SUBPATH)
        self.tags = du.load_list(self.data_dir / du.TAGS_SUBPATH)
        self.params['max_tag_idx'] = len(self.tags)
        self.params['max_word_idx'] = len(self.vocabulary)

        # Create estimator
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        config = tf.estimator.RunConfig().replace(session_config=tf.ConfigProto(gpu_options=gpu_options))
        self._estimator = tf.estimator.Estimator(model_fn, model_dir=str(model_dir), params=self.params, config=config)

        self.embedding_matrix = du.load_embeddings_matrix(self.data_dir)

    def create_train_test_hooks(self):
        return [di.create_embedding_feed_hook(self.embedding_matrix)]

    def train(self, data_path: Path):
        """
        Trains the model

        :param data_path: path to a directory containing preprocessed input files
        """
        # Create/update the parameters file
        du.store_params(self.params, self.model_dir)
        self._estimator.train(
            input_fn=di.create_input_fn(
                data_subdir=data_path,
                for_train=True,
                num_epochs=self.num_epochs,
                batch_size=self.batch_size),
            hooks=self.create_train_test_hooks())

    def evaluate(self, data_path: Path) -> dict:
        """
        Evaluates the model

        :param data_path: path to a directory containing preprocessed input files
        :return: calculated metrics dict
        """
        metrics = self._estimator.evaluate(
            input_fn=di.create_input_fn(
                data_subdir=data_path,
                for_train=False,
                num_epochs=1,
                batch_size=self.batch_size),
            hooks=self.create_train_test_hooks(),
            name='train')
        return metrics

    def predict(self, data_path: Path) -> (Iterable, str):
        """
        Creates predictions file

        :param data_path: path to the input file in raw CSV format
        :return: tuple (Iterable, str) with predictions iterable and path to a file with vocabulary extension
        """
        prediction_input = di.PredictionInput(
            data_path, self.data_dir, self.vocabulary,
            self.batch_size, self.embedding_matrix)
        predictions = self._estimator.predict(prediction_input.input_fn, hooks=prediction_input.hooks)
        return predictions, prediction_input.vocab_ext_path
