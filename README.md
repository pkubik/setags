# Stack Exchange tags prediction model

**Tensorflow** model for solving a problem specified at a [Kaggle](https://www.kaggle.com/)
problem [Transfer Learning on Stack Exchange Tags](https://www.kaggle.com/c/transfer-learning-on-stack-exchange-tags).

The pipeline assumes that there exists a data directory pointed by a `RESEARCH_DATA_DIR`
environment variable which maintains following structure:

- `$RESEARCH_DATA_DIR`
  - `stackexchange`
    - `train` (preprocessed training data in *TFRecords* format)
    - `test` (preprocessed test data in TFRecords format)
    - `raw*` ([raw CSV files from Kaggle](https://www.kaggle.com/c/transfer-learning-on-stack-exchange-tags/data))
    - `embeddings.bin*` ([Google News word2vec embeddings](https://github.com/mmihaltz/word2vec-GoogleNews-vectors))
    - `vocab.txt` (created during preprocessing step using `embeddings.bin`)
    
`*` - Entities marked with the star must be available before the preprocessing step.
    
## Usage

In order to run the preprocessing invoke following command from the repository root:
```
> python3 -m setags.data.prepare
```

Train and evaluate the module using the main module directly. You might use multiple models
by specifying names with a `-n` option. If a model with given name exists the pipeline
will further train or evaluate this model. An environment variable `TF_MODELS_DIR` might
be defined in order to store models data in specific directory. Otherwise system-specific
temporary directory will be used.
```
> python3 -m setags train -n xyz
... (training logs) ...

> python3 -m setags test -n xyz
... (evaluation report) ...
```
