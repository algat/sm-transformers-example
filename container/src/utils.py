import logging
from typing import List

from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
from transformers import EvalPrediction

logger = logging.getLogger(__name__)


def get_label_info(labels: List, task_name: str):
    label_to_id = None
    id_to_label = None
    num_labels = 1
    if task_name == "classif":
        label_list = list(set(labels))
        label_list.sort()
        label_to_id = {l: i for i, l in enumerate(label_list)}
        id_to_label = {i: l for i, l in enumerate(label_list)}
        num_labels = len(label_list)
    elif task_name == "multilabel-classif":
        label_list = list(set([l.strip() for multilabel in labels for l in multilabel.split(",")]))
        label_list.sort()
        label_to_id = {l: i for i, l in enumerate(label_list)}
        id_to_label = {i: l for i, l in enumerate(label_list)}
        num_labels = len(label_list)
    elif task_name == "ner":
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label.split())
        label_list = list(unique_labels)
        label_list.sort()
        label_to_id = {l: i for i, l in enumerate(label_list)}
        id_to_label = {i: l for i, l in enumerate(label_list)}
        num_labels = len(label_list)
    return num_labels, label_to_id, id_to_label


def preprocess_dataset(examples, 
                        tokenizer, 
                        label_to_id,
                        label_all_tokens,
                        padding, 
                        use_bbox,
                        task_name,
                        text_column_name="text", 
                        label_column_name="labels", 
                        bbox_column_name="bbox"):
    # tokenize words
    word_lists = [text.split() for text in examples[text_column_name]]
    tokenized_words = tokenizer(
        word_lists,
        padding=padding,
        truncation=True,
        is_split_into_words=True,
    )

    # Handle labels depending on task
    if task_name == "classif":
        if label_column_name in examples:
            label_id_lists = [label_to_id[l] for l in examples[label_column_name]]
        else:
            dummy_label_id = list(label_to_id.values())[0]
            label_id_lists = [dummy_label_id for _ in word_lists]
        tokenized_words["label"] = label_id_lists
    elif task_name == "multilabel-classif":
        if label_column_name in examples:
            label_id_lists = [[label_to_id[l.strip()] for l in label.split(",")] for label in examples[label_column_name]]
            one_hot_label_id_lists = [[1.0 if k in l else 0.0 for k in range(len(label_to_id))] for l in label_id_lists]
        else:
            dummy_one_hot_label_id = [0.0] * len(label_to_id)
            one_hot_label_id_lists = [dummy_one_hot_label_id for _ in word_lists]
        tokenized_words["label_ids"] = one_hot_label_id_lists
    elif task_name == "ner":
        if label_column_name in examples:
            label_lists = [label.split() for label in examples[label_column_name]]
            assert all(len(word_list) == len(label_list) for word_list, label_list in zip(word_lists, label_lists))
        else:
            dummy_label = list(label_to_id.keys())[0]
            label_lists = [[dummy_label for _ in words] for words in word_lists]
        label_id_lists = []
        for i, label in enumerate(label_lists):
            word_ids = tokenized_words.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label_to_id[label[word_idx]] if label_all_tokens else -100)
                previous_word_idx = word_idx
            label_id_lists.append(label_ids)
        tokenized_words["label"] = label_id_lists
    elif task_name == "regression":
        if label_column_name in examples:
            label_lists = [float(l) for l in examples[label_column_name]]
        else:
            label_lists = [0 for _ in word_lists]
        tokenized_words["label"] = label_lists
    
    # Handle bboxes
    if use_bbox and bbox_column_name in examples:
        # convert bbox to list of ints
        bbox_lists = list(map(lambda boxes : [[int(coord) for coord in box.split()] for box in boxes.split(",")],
                 examples[bbox_column_name]))
        assert all(len(word_list) == len(bbox_list) for word_list, bbox_list in zip(word_lists, bbox_lists))
        # format bbox
        new_bboxes = []
        for i, bbox in enumerate(bbox_lists):
            word_ids = tokenized_words.word_ids(batch_index=i)
            previous_word_idx = None
            bbox_list = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    bbox_list.append([0, 0, 0, 0])
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    bbox_list.append(bbox[word_idx])
                # For the other tokens in a word, we set the bbox to the current bbox value
                else:
                    bbox_list.append(bbox[word_idx])
                previous_word_idx = word_idx

            new_bboxes.append(bbox_list)
        tokenized_words["bbox"] = new_bboxes
    return tokenized_words


def compute_metrics_ner(p: EvalPrediction, id_to_label):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [id_to_label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id_to_label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return {
        "accuracy_score": accuracy_score(true_labels, true_predictions),
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }


def compute_metrics_regression(p: EvalPrediction):
    predictions, labels = p
    predictions = np.squeeze(predictions, axis=1)
    return {"mse": ((predictions - labels) ** 2).mean().item()}


def compute_metrics_multilabel_classif(p: EvalPrediction):
    predictions, labels = p
    predictions = 1/(1 + np.exp(-predictions)) # sigmoid
    predictions = (predictions > 0.5) # threshold
    accuracy = (predictions == labels).astype(np.float32).mean().item()
    # Maybe better : from sklearn.metrics import accuracy_score
    # accuracy = accuracy_score(labels, true_predictions)
    return {"accuracy": accuracy}


def compute_metrics_classif(p: EvalPrediction):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": (predictions == labels).astype(np.float32).mean().item()}


def compute_metrics(p, id_to_label, task_name: str):
    if task_name == "regression":
        return compute_metrics_regression(p)
    elif task_name == "classif":
        return compute_metrics_classif(p)
    elif task_name == "multilabel-classif":
        return compute_metrics_multilabel_classif(p)
    elif task_name == "ner":
        return compute_metrics_ner(p, id_to_label)
