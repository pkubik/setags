import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


MIN_AFTER_DEQUEUE = 512
PARAMS_FILENAME = 'params.json'
MODELS_DIR_VARIABLE_NAME = 'TENSORFLOW_MODELS_DIR'
DATA_DIR_VARIABLE_NAME = 'RESEARCH_DATA_DIR'
PROBLEM_SUBDIR = 'stackexchange'
TRAIN_DATA_SUBDIR = 'train'
TEST_DATA_SUBDIR = 'test'
RAW_DATA_SUBDIR = 'raw'
EMBEDDINGS_SUBPATH = 'embeddings.bin'
VOCAB_SUBPATH = 'vocab.txt'
UNKNOWN_WORD_CODE = -1


def load_vocabulary(data_dir: Path) -> list:
    """
    Loads the vocabulary. If it does not exist, then it is created from the embeddings.
    A word with index `0` is the padding/EOS symbol `</s>`.

    :param data_dir: path to the problem's data directory
    :return: list of words
    """
    vocab_path = data_dir / VOCAB_SUBPATH
    vocab = load_list(vocab_path)
    if len(vocab) > 0:
        return vocab

    logger.warning("Vocabulary file not found. Loading embeddings to generate the vocabulary file.")
    import gensim
    embeddings_path = data_dir / EMBEDDINGS_SUBPATH
    embeddings = gensim.models.KeyedVectors.load_word2vec_format(str(embeddings_path), binary=True)
    vocab = embeddings.index2word
    store_list(vocab, vocab_path)
    return vocab


def get_data_dir() -> Path:
    import tempfile

    data_dir = os.environ.get(DATA_DIR_VARIABLE_NAME)
    if data_dir is not None:
        logger.info("Using '{}' as the data directory.".format(data_dir))
        data_dir = Path(data_dir) / PROBLEM_SUBDIR
    else:
        data_dir = Path(tempfile.gettempdir()) / 'data'
        logger.warning("Data directory variable '{}' not defined. Using '{}' in its place.".format(
            DATA_DIR_VARIABLE_NAME, data_dir))
        data_dir = data_dir / PROBLEM_SUBDIR

    return data_dir


def get_model_dir(name: str = None) -> Path:
    import tempfile

    models_dir = os.environ.get(MODELS_DIR_VARIABLE_NAME)
    if models_dir is None:
        tmp_dir = Path(tempfile.gettempdir())
        default_models_subdir = 'tf_models'
        models_dir = tmp_dir / default_models_subdir
    else:
        models_dir = Path(models_dir)

    if name is not None:
        model_dir = models_dir / name
    else:
        unnamed_models_dir = models_dir / 'unnamed'
        os.makedirs(str(unnamed_models_dir), exist_ok=True)
        model_dir = Path(tempfile.mkdtemp(dir=str(unnamed_models_dir), prefix=''))

    return model_dir


def store_params(params: dict, model_dir: Path):
    os.makedirs(str(model_dir), exist_ok=True)
    with (model_dir / PARAMS_FILENAME).open('w') as params_file:
        json.dump(params, params_file)


def load_params(model_dir: Path) -> dict:
    os.makedirs(str(model_dir), exist_ok=True)
    try:
        with (model_dir / PARAMS_FILENAME).open() as params_file:
            return json.load(params_file)
    except FileNotFoundError:
        return {}


def store_list(values: list, path: Path):
    with path.open('w') as output_file:
        for value in values:
            output_file.write(str(value) + '\n')


def load_list(path: Path) -> list:
    try:
        with path.open() as input_file:
            return [value.strip() for value in input_file.readlines()]
    except FileNotFoundError:
        return []
