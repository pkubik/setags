from enum import Enum
from functools import partial
from pathlib import Path

import tensorflow as tf

import setags.data.utils as du
import setags.data.input as di
from setags.cli import CLI
from setags.model import model_fn, DEFAULT_PARAMS


class Action(Enum):
    TRAIN = 'train'
    TEST = 'test'


def run(action: Action, model_dir: Path, overrides: dict):
    print("Using a model from '{}' ({})".format(model_dir, action.value))

    params = {}
    params.update(DEFAULT_PARAMS)
    params.update(du.load_params(model_dir))
    params.update(overrides)

    num_epochs = params['num_epochs']
    batch_size = params['batch_size']

    data_dir = du.get_data_dir()
    train_dir = data_dir / du.TRAIN_DATA_SUBDIR
    test_dir = data_dir / du.TEST_DATA_SUBDIR

    tags = du.load_list(data_dir / du.TAGS_SUBPATH)
    params['max_tag_idx'] = len(tags)
    create_input_fn = partial(di.create_input_fn, batch_size=batch_size, data_dir=data_dir)

    # Create estimator
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    config = tf.estimator.RunConfig().replace(session_config=tf.ConfigProto(gpu_options=gpu_options))
    e = tf.estimator.Estimator(model_fn, model_dir=str(model_dir), params=params, config=config)

    # Train model
    if action == Action.TRAIN:
        # Create/update the parameters file
        du.store_params(params, model_dir)
        e.train(input_fn=create_input_fn(data_subdir=train_dir, for_train=True, num_epochs=num_epochs))

    # Evaluate model
    if action in [Action.TRAIN, Action.TEST]:
        train_metrics = e.evaluate(input_fn=create_input_fn(data_subdir=train_dir, for_train=False))
        print('Train set metrics:\n{}'.format(train_metrics))
        test_metrics = e.evaluate(input_fn=create_input_fn(data_subdir=test_dir, for_train=False))
        print('Test set metrics:\n{}'.format(test_metrics))


def main():
    allowed_actions = ['train', 'test']
    allowed_params = sorted(DEFAULT_PARAMS.keys())
    cli = CLI(allowed_actions, allowed_params)
    run(Action(cli.action), cli.model_dir, cli.overrides)


if __name__ == '__main__':
    main()
