![Build](https://codebuild.eu-central-1.amazonaws.com/badges?uuid=eyJlbmNyeXB0ZWREYXRhIjoic3AwRG9wSVRVZ1hiaytvUVlTQVNQbEdNVDdyaDYzMkJZY3dRZGdzNDAzSUdVWUpQaXhzUkx2RjBQZ093cTQ3UEkvVW52Y3NCZ1dqYkU4UGtmL0JiUVRzPSIsIml2UGFyYW1ldGVyU3BlYyI6IjE3dUcyN0ZsR3FSZkMyTjkiLCJtYXRlcmlhbFNldFNlcmlhbCI6MX0%3D&branch=master)

# SageMaker container for fine-tuning and serving of Transformer models

Available task:
* Text classification
* Token classification (NER)
* Regression

Tested on:
* Bert models
* Distilbert
* Longformer models
* LayoutLM model (only available for Token classification)

## Usage for Training

# Training data format

To use the model, your training data should consist of 3 csv (with tab separator `\t`) files : `train.csv`, `dev.csv` and `test.csv` located in a directory on s3, each one having the same format.

For token classification task, the format should be:

| text        | bbox           | labels  |
| ------------- |:-------------:| -----:|
| Hi, I am Rev. And you?   | 47 253 60 277, 45 205 60 254, 50 400 78 436, 100 207 146 567 50 400 78 436, 100 207 146 567 | O O O B-PER O O |
| Lorem ipsum dolor sit amet,      | 879 253 899 277, 45 205 60 254, 50 400 78 436, 100 207 146 567, 100 207 146 567 | O O O B-PER I-PER O |
| Quisque justo  | 50 400 78 436, 100 207 146 567 | O O |
|...|...|...|



```
labels	text	bbox
file_folder	TEE ETOS TENNYSON TENN ... Ys Men EHOT Hotce 1	47 253 60 277,45 205 60 254,...,47 134 64 206,46 164 60 202
presentation	FOR IMMEDIATE RELEASE CONTACT: ... Scott Williams	150 102 188 114,191 101 301 114,302 101 387 114,...,599 101 694 114
presentation	JUN 26 195 02:22PM ... P.5/13 Good afternoon	160 102 188 114,191 101 301 114,302 101 387 114,...,599 101 694 114
news_article	"hwc ses SG, 2/+ H61 LCc us 6 TLuJ ... July 25, 1983 Coure 2	191 130 249 138,257 130 275 139,...,282 132 290 136
...
```



* tab separator
* columns with headers: `text`, `labels` and optionally `bbox`
* `text` column
Each one having the same format: tab separ


# Hyperparameters

