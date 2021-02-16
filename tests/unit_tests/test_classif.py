import os
import shutil
import pytest
import importlib

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.path.join(REPO_ROOT, "cache_dir") # use the cache dir of the repo

# dirs related to test output
TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")
CHECKPOINT_DIR = os.path.join(TEST_DATA_DIR, "checkpoints")
OUTPUT_MODEL_DIR = os.path.join(TEST_DATA_DIR, "output_dir")
LOGGING_DIR = os.path.join(TEST_DATA_DIR, "logging_dir")

# test dataset dir
DATASET_DIR = os.path.join(TEST_DATA_DIR, "dataset_classif")

@pytest.fixture
def teardown_cleaning():
    yield
    dirpaths = [CHECKPOINT_DIR, OUTPUT_MODEL_DIR, LOGGING_DIR]
    for dirpath in dirpaths:
       if os.path.exists(dirpath) and os.path.isdir(dirpath):
           shutil.rmtree(dirpath)

bert_args = {'output_dir': CHECKPOINT_DIR,
            'overwrite_output_dir': True,
            'do_train': True,
            'do_eval': True,
            'do_predict': True,
            'evaluation_strategy': 'steps',
            'per_device_train_batch_size': 5,
            'per_device_eval_batch_size': 5,
            'max_steps': 11,
            'logging_dir': LOGGING_DIR,
            'logging_steps': 5,
            'save_steps': 11,
            'remove_unused_columns': True,
            'load_best_model_at_end': True,
            'metric_for_best_model': 'eval_loss',
            'model_name_or_path': 'bert-base-uncased',
            'cache_dir': CACHE_DIR,
            'task_name': 'classif',
            'train_file': os.path.join(DATASET_DIR, "train.csv"),
            'validation_file': os.path.join(DATASET_DIR, "dev.csv"),
            'test_file': os.path.join(DATASET_DIR, "test.csv"),
            'pad_to_max_length': False,
            'use_bbox': False,
            'sagemaker_output_path': OUTPUT_MODEL_DIR}

longformer_args = {'output_dir': CHECKPOINT_DIR,
            'overwrite_output_dir': True,
            'do_train': True,
            'do_eval': True,
            'do_predict': True,
            'evaluation_strategy': 'steps',
            'per_device_train_batch_size': 2,
            'per_device_eval_batch_size': 2,
            'max_steps': 3,
            'logging_dir': LOGGING_DIR,
            'logging_steps': 2,
            'save_steps': 2,
            'remove_unused_columns': True,
            'load_best_model_at_end': True,
            'metric_for_best_model': 'eval_loss',
            'model_name_or_path': 'allenai/longformer-base-4096',
            'cache_dir': CACHE_DIR,
            'task_name': 'classif',
            'train_file': os.path.join(DATASET_DIR, "train.csv"),
            'validation_file': os.path.join(DATASET_DIR, "dev.csv"),
            'test_file': os.path.join(DATASET_DIR, "test.csv"),
            'pad_to_max_length': False,
            'use_bbox': False,
            'sagemaker_output_path': OUTPUT_MODEL_DIR}

distilbert_args = {'output_dir': CHECKPOINT_DIR,
            'overwrite_output_dir': True,
            'do_train': True,
            'do_eval': True,
            'do_predict': True,
            'evaluation_strategy': 'steps',
            'per_device_train_batch_size': 5,
            'per_device_eval_batch_size': 5,
            'max_steps': 11,
            'logging_dir': LOGGING_DIR,
            'logging_steps': 5,
            'save_steps': 5,
            'remove_unused_columns': True,
            'load_best_model_at_end': True,
            'metric_for_best_model': 'eval_loss',
            'model_name_or_path': 'distilbert-base-cased',
            'cache_dir': CACHE_DIR,
            'task_name': 'classif',
            'train_file': os.path.join(DATASET_DIR, "train.csv"),
            'validation_file': os.path.join(DATASET_DIR, "dev.csv"),
            'test_file': os.path.join(DATASET_DIR, "test.csv"),
            'pad_to_max_length': False,
            'use_bbox': False,
            'sagemaker_output_path': OUTPUT_MODEL_DIR}

args_list = [bert_args , longformer_args, distilbert_args]

@pytest.mark.parametrize("args", args_list)
def test_training(args, teardown_cleaning):
    # test training
    from container.src.train_model import train_model
    result = train_model(args)

    keys = ['eval_loss', 'eval_accuracy', 'eval_runtime', 'eval_samples_per_second', 'epoch']
    assert all(k in result for k in keys)

    # check args files saved
    assert os.path.isfile(os.path.join(OUTPUT_MODEL_DIR, "model_args.bin"))
    assert os.path.isfile(os.path.join(OUTPUT_MODEL_DIR, "data_args.bin"))
    assert os.path.isfile(os.path.join(OUTPUT_MODEL_DIR, "training_args.bin"))
    
    # check model files saved
    assert os.path.isfile(os.path.join(OUTPUT_MODEL_DIR, "pytorch_model.bin"))
    assert os.path.isfile(os.path.join(OUTPUT_MODEL_DIR, "trainer_state.json"))
    assert os.path.isfile(os.path.join(OUTPUT_MODEL_DIR, "config.json"))
    assert os.path.isfile(os.path.join(OUTPUT_MODEL_DIR, "special_tokens_map.json"))
    assert (os.path.isfile(os.path.join(OUTPUT_MODEL_DIR, "vocab.txt")) or
            os.path.isfile(os.path.join(OUTPUT_MODEL_DIR, "vocab.json")))  

    # check eval / pred files are saved
    assert os.path.isfile(os.path.join(OUTPUT_MODEL_DIR, "eval_results.txt"))
    assert os.path.isfile(os.path.join(OUTPUT_MODEL_DIR, "test_predictions.txt"))
    assert os.path.isfile(os.path.join(OUTPUT_MODEL_DIR, "test_results.txt"))
    assert os.path.isfile(os.path.join(OUTPUT_MODEL_DIR, "train_results.txt"))

    # add resume training test ?
    
    # test predictions
    import container.src.inference
    importlib.reload(container.src.inference)
    from container.src.inference import ScoringService
    ScoringService.get_model(OUTPUT_MODEL_DIR)
    input = {'text': ['hello I am ALexis', "how are you", "are you doing fine??"],
           'bbox': ["12 34 87 40, 12 34 87 40, 12 34 87 40, 12 34 87 40", "12 34 87 40, 12 34 87 40, 12 34 87 40",
                   "12 34 87 40, 12 34 87 40, 12 34 87 40, 12 34 87 40"]}
    result = ScoringService.predict(OUTPUT_MODEL_DIR, input)
    #result = {"pred": [], "proba":[]}
    assert len(result["pred"]) == len(input["text"])
    assert len(result["proba"]) == len(input["text"])
    assert all(0 <= p <= 1 for p in result["proba"])