![Build](https://codebuild.eu-central-1.amazonaws.com/badges?uuid=eyJlbmNyeXB0ZWREYXRhIjoic3AwRG9wSVRVZ1hiaytvUVlTQVNQbEdNVDdyaDYzMkJZY3dRZGdzNDAzSUdVWUpQaXhzUkx2RjBQZ093cTQ3UEkvVW52Y3NCZ1dqYkU4UGtmL0JiUVRzPSIsIml2UGFyYW1ldGVyU3BlYyI6IjE3dUcyN0ZsR3FSZkMyTjkiLCJtYXRlcmlhbFNldFNlcmlhbCI6MX0%3D&branch=master)

# SageMaker container for fine-tuning and serving of Transformer models

**Available task:** Text classification + Token classification (NER)
**Tested on:** Bert, Distilbert, Longformer, LayoutLM

## Usage for Training

### Training data format

To use the model, your training data should consist of 3 csv (with tab separator `\t`) files : `train.csv`, `dev.csv` and `test.csv` located in a directory on s3, each one having the same format.

* For token classification task, the format should be:

| text        |  labels  |
| ------------- |:-------------:|
| Hi, I am Rev. And? | O O O B-PER O |
| Lorem ipsum dolor sit amet, | O O O B-PER I-PER O |
| Quisque justo | O O |
|...|...|

> labels in this case are word labels (one label for each word). Words in text are separated by space. So in the example, "Hi," has label "O".
> An example of these files can be found in the [tests](./tests/unit_tests/dataset_classif/)

* For text classification:

| text        |  labels  |
| ------------- |:-------------:|
| Hi, I am Rev. And? | "My category" |
| Lorem ipsum dolor sit amet, | "other" |
| Quisque justo | "category2" |
|...|...|...|

> An example of these files can be found in the [tests](./tests/unit_tests/dataset_ner/)

* Optionaly, you can add a column `bbox` (used by LayoutLM model) containing the coordinates of the word in the document.

| text        | bbox           | labels  |
| ------------- |:-------------:| -----:|
| Hi, I am Rev. And?   | 47 253 60 277, 45 205 60 254, 50 400 78 436, 100 207 146 567 50 400 78 436 | O O O B-PER O |
| Lorem ipsum dolor sit amet,      | 879 253 899 277, 45 205 60 254, 50 400 78 436, 100 207 146 567, 100 207 146 567 | O O O B-PER I-PER O |
| Quisque justo  | 50 400 78 436, 100 207 146 567 | O O |
|...|...|...|

> Coordinates for one word are x0 y0 x1 y1, which are relative position (origin top left corner), normalized to be between 1 and 1000. Refer to LayoutLM paper for more information.

### Hyperparameters

