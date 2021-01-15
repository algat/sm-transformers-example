import os
import shutil
import pytest
import subprocess
import requests
import json
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.path.join(REPO_ROOT, "cache_dir") # use the cache dir of the repo

# dirs related to test
TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")
CHECKPOINT_DIR = os.path.join(TEST_DATA_DIR, "checkpoints")
OUTPUT_MODEL_DIR = os.path.join(TEST_DATA_DIR, "model")
LOGGING_DIR = os.path.join(TEST_DATA_DIR, "output")
#DATASET_DIR = os.path.join(TEST_DATA_DIR, "dataset")


@pytest.fixture
def docker_fixture():
    if not os.path.exists(LOGGING_DIR):
        os.makedirs(LOGGING_DIR)
    yield 
    dirpaths = [CHECKPOINT_DIR, OUTPUT_MODEL_DIR, LOGGING_DIR]
    for dirpath in dirpaths:
        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            shutil.rmtree(dirpath)

def test_docker(docker_fixture):
    # build the image
    bashCommand = "docker build -t sm-transformer -f Dockerfile.cpu {}".format(REPO_ROOT)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    assert process.returncode == 0

    # check training
    bashCommand = "docker run -v {}:/opt/ml -v {}:/tmp --rm sm-transformer sh -c train".format(TEST_DATA_DIR, CACHE_DIR)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    assert process.returncode == 0
    assert os.path.isfile(os.path.join(OUTPUT_MODEL_DIR, "model_args.bin"))
    assert os.path.isfile(os.path.join(OUTPUT_MODEL_DIR, "data_args.bin"))
    assert os.path.isfile(os.path.join(OUTPUT_MODEL_DIR, "training_args.bin"))
    assert os.path.isfile(os.path.join(OUTPUT_MODEL_DIR, "pytorch_model.bin"))
    assert os.path.isfile(os.path.join(OUTPUT_MODEL_DIR, "trainer_state.json"))
    assert os.path.isfile(os.path.join(OUTPUT_MODEL_DIR, "config.json"))
    assert os.path.isfile(os.path.join(OUTPUT_MODEL_DIR, "special_tokens_map.json"))
    assert (os.path.isfile(os.path.join(OUTPUT_MODEL_DIR, "vocab.txt")) or
            os.path.isfile(os.path.join(OUTPUT_MODEL_DIR, "vocab.json")))  
    assert os.path.isfile(os.path.join(OUTPUT_MODEL_DIR, "eval_results.txt"))
    assert os.path.isfile(os.path.join(OUTPUT_MODEL_DIR, "test_predictions.txt"))
    assert os.path.isfile(os.path.join(OUTPUT_MODEL_DIR, "test_results.txt"))
    assert os.path.isfile(os.path.join(OUTPUT_MODEL_DIR, "train_results.txt"))

    # check resume training capability
    shutil.rmtree(OUTPUT_MODEL_DIR) # delete outputdir
    bashCommand = "docker run -v {}:/opt/ml -v {}:/tmp --rm sm-transformer sh -c train".format(TEST_DATA_DIR, CACHE_DIR)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate() #blocking operation
    assert process.returncode == 0
    assert os.path.isfile(os.path.join(OUTPUT_MODEL_DIR, "pytorch_model.bin"))
    # additionaly, we could check that model_args["model_name_or_path"] was a checkpoint dir

    # serve model locally
    bashCommand = "docker run -v {}:/opt/ml -p 8080:8080 --rm sm-transformer serve".format(TEST_DATA_DIR)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    time.sleep(5)
    assert process.poll() == None # check still running

    # check ping
    r = requests.get(url = 'http://localhost:8080/ping')
    assert r.status_code == 200

    # check prediction invocation
    input_data = {'text': ['hello I am ALexis', "how are you", "are you doing fine??"],
                'bbox': ["12 34 87 40, 12 34 87 40, 12 34 87 40, 12 34 87 40", "12 34 87 40, 12 34 87 40, 12 34 87 40",
                        "12 34 87 40, 12 34 87 40, 12 34 87 40, 12 34 87 40"]}
    r = requests.post(url = 'http://localhost:8080/invocations',
                data = json.dumps(input_data),
                headers = {'Content-Type': 'application/json'})
    assert r.status_code == 200
    result = r.json()
    assert "predictions" in result
    assert len(result["predictions"]) == len(input_data["text"])

    # end process todo: SHOULD BE IN FIXTURE !!!!!
    process.terminate()