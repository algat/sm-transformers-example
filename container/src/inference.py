import os
import logging
import numpy as np
import torch

from datasets import Dataset

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    default_data_collator,
)

from .utils import preprocess_dataset

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.
class ScoringService(object):
    model = None                # Where we keep the model when it's loaded
    config = None
    data_args = None
    tokenizer = None

    @classmethod
    def get_model(cls, path_to_model):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            logger.info("Loading Model from %s", path_to_model)
            
            # check args file are here
            assert os.path.isfile(os.path.join(path_to_model, "model_args.bin"))
            assert os.path.isfile(os.path.join(path_to_model, "data_args.bin"))
            assert os.path.isfile(os.path.join(path_to_model, "training_args.bin"))
            
            # check model files are here
            assert os.path.isfile(os.path.join(path_to_model, "pytorch_model.bin"))
            assert os.path.isfile(os.path.join(path_to_model, "config.json"))
            assert os.path.isfile(os.path.join(path_to_model, "special_tokens_map.json"))
            assert (os.path.isfile(os.path.join(path_to_model, "vocab.txt")) or
                    os.path.isfile(os.path.join(path_to_model, "vocab.json")))  
            
            # load args used during training
            model_args = torch.load(os.path.join(path_to_model, "model_args.bin")) # a dict
            data_args = torch.load(os.path.join(path_to_model, "data_args.bin")) # a dict
            training_args = torch.load(os.path.join(path_to_model, "training_args.bin")) # a TrainingArguments object
            
            # load trained model
            config = AutoConfig.from_pretrained(path_to_model)
            tokenizer = AutoTokenizer.from_pretrained(path_to_model)
            if data_args["task_name"] == "ner":
                model = AutoModelForTokenClassification.from_pretrained(path_to_model)
            else:
                model = AutoModelForSequenceClassification.from_pretrained(path_to_model)
                
            # data collator
            if data_args["pad_to_max_length"]:
                data_collator = default_data_collator
            else:
                if data_args["task_name"] == "ner":
                    data_collator = DataCollatorForTokenClassification(tokenizer)
                else:
                    data_collator = None # will default to DataCollatorWithPadding

            # set trainer (only used for prediction though)
            trainer = Trainer(
                model=model,
                args=training_args,
                tokenizer=tokenizer,
                data_collator=data_collator,
            )
            
            cls.model = trainer
            cls.config = config
            cls.data_args = data_args
            cls.tokenizer = tokenizer
        return cls.model, cls.config, cls.data_args, cls.tokenizer

    @classmethod
    def predict(cls, path_to_model, input):
        """For the input, do the predictions and return them.
        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""        
        
        trainer, config, data_args, tokenizer = cls.get_model(path_to_model)

        text_column_name = "text"
        label_column_name = "labels"
        bbox_columns_name = "bbox"
        pred_dataset = Dataset.from_dict(input)
        tokenized_datasets = pred_dataset.map(
            lambda x: preprocess_dataset(x, 
                                        tokenizer, 
                                        config.label2id, 
                                        data_args["label_all_tokens"], 
                                        "max_length" if data_args["pad_to_max_length"] else False, 
                                        data_args["use_bbox"],
                                        data_args["task_name"]),
            batched=True,
            num_proc=data_args["preprocessing_num_workers"],
            load_from_cache_file=not data_args["overwrite_cache"],
        )
        logger.info("Datasets %s", tokenized_datasets)
        logger.info("Column names %s", tokenized_datasets.column_names)
        logger.info("Sample example %s", tokenized_datasets[0])
        
        # Get predictions
        true_predictions = None
        predictions, labels, _ = trainer.predict(tokenized_datasets, metric_key_prefix="pred")
        if data_args["task_name"] == "classif":
            true_predictions = [config.id2label[p] for p in np.argmax(predictions, axis=1)]
        elif data_args["task_name"] == "regression":
            true_predictions = np.squeeze(predictions)
        elif data_args["task_name"] == "ner":
            predictions = np.argmax(predictions, axis=2)
            true_predictions = [
                [config.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
        logger.info("true_predictions %s", true_predictions)
        return true_predictions
