![Build](https://codebuild.eu-central-1.amazonaws.com/badges?uuid=eyJlbmNyeXB0ZWREYXRhIjoic3AwRG9wSVRVZ1hiaytvUVlTQVNQbEdNVDdyaDYzMkJZY3dRZGdzNDAzSUdVWUpQaXhzUkx2RjBQZ093cTQ3UEkvVW52Y3NCZ1dqYkU4UGtmL0JiUVRzPSIsIml2UGFyYW1ldGVyU3BlYyI6IjE3dUcyN0ZsR3FSZkMyTjkiLCJtYXRlcmlhbFNldFNlcmlhbCI6MX0%3D&branch=master)

# Fine-tuning and serving Transformer models on Sagemaker

**Available task:** Text classification (multiclass & multilabel), Token classification (NER)

**Tested on:** Bert, Distilbert, Longformer, LayoutLM\*

\* <sup>LayoutLM is only available for Token classification</sup>

## Usage for Training

### Training data format

To use the model, your training data should consist of 3 tab separated (`\t`) csv files : `train.csv`, `dev.csv` and `test.csv` located in a directory on s3, each one having the same format.

**For token classification task**

| text        |  labels  |
| ------------- |:-------------:|
| Hi, I am Rev. And? | O O O B-PER O |
| Lorem ipsum dolor sit amet, | O O O B-PER I-PER O |
| Quisque justo | O O |
|...|...|

> labels in this case are word labels (one label for each word). Words in text are separated by space. So in the example, "Hi," has label "O".
> An example of these files can be found in the [tests](./tests/unit_tests/test_data/dataset_classif/)

**For text classification (Multiclass)**

| text        |  labels  |
| ------------- |:-------------:|
| Hi, I am Rev. And? | "My category" |
| Lorem ipsum dolor sit amet, | "other" |
| Quisque justo | "category2" |
|...|...|...|

> An example of these files can be found in the [tests](./tests/unit_tests/test_data/dataset_ner/)

**For text classification (Multilabel)**

| text        |  labels  |
| ------------- |:-------------:|
| Hi, I am Rev. And? | "My category","other" |
| Lorem ipsum dolor sit amet, | "other" |
| Quisque justo | "category2","My category" |
|...|...|...|

> An example of these files can be found in the [tests](./tests/unit_tests/test_data/dataset_multilabel_classif/)

**Optionaly for LayoutLM**, you can add a column `bbox` containing the coordinates of the word in the document.

| text        | bbox           | labels  |
| ------------- |:-------------:| -----:|
| Hi, I am Rev. And?   | 47 253 60 277, 45 205 60 254, 50 400 78 436, 100 207 146 567 50 400 78 436 | O O O B-PER O |
| Lorem ipsum dolor sit amet,      | 879 253 899 277, 45 205 60 254, 50 400 78 436, 100 207 146 567, 100 207 146 567 | O O O B-PER I-PER O |
| Quisque justo  | 50 400 78 436, 100 207 146 567 | O O |
|...|...|...|

> Coordinates for one word are x0 y0 x1 y1, which are relative position (origin top left corner), normalized to be between 1 and 1000. Refer to LayoutLM paper for more information.

### Hyperparameters

You can provide SageMaker Hyperparameters to adapt the training parameters:

* `task_name`: Either "ner", "classif", "multilabel-classif" or "regression"
* `model_name`: pretrained Transformer model to use (e.g. "bert-base-uncased", "allenai/longformer-base-4096", "microsoft/layoutlm-base-uncased")
* `max_steps`: Number of training steps (e.g. "1000")
* `use_bbox`: Whether to use the bbox column (if provided in training data). Should be "true" if LayoutLM.
* `per_device_train_batch_size`: Batch size per GPU/CPU for training (e.g. "5")
* `per_device_eval_batch_size`: Batch size per GPU/CPU for training (e.g. "5")

**NB:** Recommended hyperparameters when using GPU instance `ml.g4dn.xlarge` or `ml.g4dn.2xlarge` (one T4 gpu with 16 GB ram):

| &nbsp; | `per_device_train_batch_size` / `per_device_eval_batch_size` |
| ------------- |:-------------:|
| Bert, Distilbert, LayoutLM |  10 |
| Longformer |  1 |

## Usage for Serving

Whether using Endpoint or Batch Transform, there are two options for the input/output data format: `application/json` and `application/jsonlines`.

### Input data format

* With Content-type `application/json`:

You can provide one or multiple samples with this json structure:

```json
{"data":[{"text": "this is an example text"}, 
         {"text": "how are you"},
         {"text": "are you doing fine?"}]}
```

* With Content-type `application/jsonlines`:

You can provide one or multiple samples with this jsonlines structure:

```json
{"text": "this is an example text"}\n
{"text": "how are you"}\n
{"text": "are you doing fine?"}
```

### Ouput data format

* With Accept `application/json`, the response will be (for classification task):

```json
{"predictions": [{"pred": "tech", "proba": 0.553991973400116}, 
                 {"pred": "politics", "proba": 0.3334293067455292}, 
                 {"pred": "tech", "proba": 0.5057740211486816}]}
```

* With Accept `application/jsonlines`, the response will be:

```json
{"pred": "tech", "proba": 0.553991973400116}\n
{"pred": "politics", "proba": 0.3334293067455292}\n
{"pred": "tech", "proba": 0.5057740211486816}
```

**NB:** For Batch Transform, we recommend using `jsonlines` and/or the following strategies:

| ContentType | Recommended SplitType |
| --- | --- |
| application/jsonlines | Line |
| application/json | None |


| Accept | Recommended AssembleWith |
| --- | --- |
| application/jsonlines | Line |
| application/json | None |


## Example notebook

And example notebook to train/serve with SageMaker is available [here](./examples/train_model_with_sm.ipynb)

## Running tests

Unit tests: 

```python
python -m pytest tests/unit_tests -v -s -o log_cli=true -o log_cli_level="INFO"
```

Docker tests:

```python
python -m pytest tests/docker_test -v -s -o log_cli=true -o log_cli_level="INFO"
```

**NB:** The first run of the tests will take more time than the subsequent ones since some pre-trained models will be downloaded.