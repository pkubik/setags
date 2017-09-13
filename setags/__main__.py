import tempfile
from enum import Enum
from functools import partial
from pathlib import Path

import tensorflow as tf

import setags.data.input as di
import setags.data.utils as du
from setags.cli import CLI
from setags.utils import cprint
from setags.logging import setup_logger
from setags.model import model_fn, DEFAULT_PARAMS

PREDICTION_DATA_FILENAME = 'test.csv'


class Action(Enum):
    TRAIN = 'train'
    TEST = 'test'
    PREDICT = 'predict'


def run(action: Action, model_dir: Path, overrides: dict):
    cprint("Using a model from '{}' ({})".format(model_dir, action.value))

    params = {}
    params.update(DEFAULT_PARAMS)
    params.update(du.load_params(model_dir))
    params.update(overrides)

    num_epochs = params['num_epochs']
    batch_size = params['batch_size']

    data_dir = du.get_data_dir()
    train_dir = data_dir / du.TRAIN_DATA_SUBDIR
    test_dir = data_dir / du.TEST_DATA_SUBDIR

    vocabulary = du.load_list(data_dir/ du.VOCAB_SUBPATH)
    tags = du.load_list(data_dir / du.TAGS_SUBPATH)
    params['max_tag_idx'] = len(tags)
    params['max_word_idx'] = len(vocabulary)
    create_input_fn = partial(di.create_input_fn, batch_size=batch_size)

    # Create estimator
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    config = tf.estimator.RunConfig().replace(session_config=tf.ConfigProto(gpu_options=gpu_options))
    e = tf.estimator.Estimator(model_fn, model_dir=str(model_dir), params=params, config=config)

    hooks = []
    if action in [Action.TRAIN, Action.TEST]:
        embedding_matrix = du.load_embeddings_matrix(data_dir)
        hooks = [di.create_embedding_feed_hook(embedding_matrix)]

    # Train model
    if action == Action.TRAIN:
        # Create/update the parameters file
        du.store_params(params, model_dir)
        e.train(input_fn=create_input_fn(data_subdir=train_dir, for_train=True, num_epochs=num_epochs), hooks=hooks)

    # Evaluate model
    if action in [Action.TRAIN, Action.TEST]:
        train_metrics = e.evaluate(input_fn=create_input_fn(data_subdir=train_dir, for_train=False), hooks=hooks)
        cprint('Train set metrics:\n{}'.format(train_metrics))
        test_metrics = e.evaluate(input_fn=create_input_fn(data_subdir=test_dir, for_train=False), hooks=hooks)
        cprint('Test set metrics:\n{}'.format(test_metrics))

    # Make predictions
    if action == Action.PREDICT:
        prediction_data_path = data_dir / du.RAW_DATA_SUBDIR / PREDICTION_DATA_FILENAME
        prediction_input = di.PredictionInput(prediction_data_path, data_dir, vocabulary, batch_size)
        predictions = e.predict(prediction_input.input_fn, hooks=prediction_input.hooks)
        cprint("Storing new words in '{}'".format(prediction_input.vocab_ext_path))

        with tempfile.NamedTemporaryFile(mode='w+t', prefix='tags-', delete=False) as predictions_file:
            cprint("Storing tagging output in '{}'".format(predictions_file.name))
            predictions_file.write('id,tags\n')
            for p in predictions:
                predictions_file.write('{},{}\n'.format(p['id'], tags[p['tags']]))


def main():
    setup_logger()
    allowed_actions = [a.value for a in Action]
    allowed_params = sorted(DEFAULT_PARAMS.keys())
    cli = CLI(allowed_actions, allowed_params)
    run(Action(cli.action), cli.model_dir, cli.overrides)


if __name__ == '__main__':
    main()
