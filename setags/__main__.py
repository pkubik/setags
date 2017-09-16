import tempfile
from enum import Enum
from pathlib import Path
import setags.data.utils as du
from setags.cli import CLI
from setags.estimator import Estimator
from setags.utils import cprint
from setags.logging import setup_logger
from setags.model import DEFAULT_PARAMS

PREDICTION_DATA_FILENAME = 'test.csv'


class Action(Enum):
    TRAIN = 'train'
    TEST = 'test'
    PREDICT = 'predict'


def run(action: Action, model_dir: Path, overrides: dict):
    cprint("Using a model from '{}' ({})".format(model_dir, action.value))

    data_dir = du.get_data_dir()
    train_dir = data_dir / du.TRAIN_DATA_SUBDIR
    test_dir = data_dir / du.TEST_DATA_SUBDIR

    e = Estimator(model_dir, data_dir, overrides)

    # Train model
    if action == Action.TRAIN:
        e.train(train_dir)

    # Evaluate model
    if action in [Action.TRAIN, Action.TEST]:
        train_metrics = e.evaluate(train_dir)
        cprint('Train set metrics:\n{}'.format(train_metrics))
        test_metrics = e.evaluate(test_dir)
        cprint('Test set metrics:\n{}'.format(test_metrics))

    # Make predictions
    if action == Action.PREDICT:
        prediction_data_path = data_dir / du.RAW_DATA_SUBDIR / PREDICTION_DATA_FILENAME
        predictions, vocab_ext_path = e.predict(prediction_data_path)
        cprint("Storing new words in '{}'".format(vocab_ext_path))

        with tempfile.NamedTemporaryFile(mode='w+t', prefix='tags-', delete=False) as predictions_file:
            cprint("Storing tagging output in '{}'".format(predictions_file.name))
            predictions_file.write('id,tags\n')
            for p in predictions:
                predictions_file.write('{},{}\n'.format(p['id'], e.tags[p['tags']]))


def main():
    setup_logger()
    allowed_actions = [a.value for a in Action]
    allowed_params = sorted(DEFAULT_PARAMS.keys())
    cli = CLI(allowed_actions, allowed_params)
    run(Action(cli.action), cli.model_dir, cli.overrides)


if __name__ == '__main__':
    main()
