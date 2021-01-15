# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import json
import flask
import logging

from src.inference import ScoringService
logger = logging.getLogger(__name__)

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model(model_path) is not None  # You can insert a health check here
    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a SINGLE sample data, though the text can be long :) 
    Formats allowed are text/plain text and application/json.
    Input
        - plain text: "Hello I want a NER prediction"
        - application/json: {"text": "Hello I want a NER prediction"}
    Output
        - application/json: {'result': 'Hello O\nI O\nwant O\na O\nNER O\nprediction O\n'}
    """
    text = None
    data = None

    # Parse input data based on content-type received
    #if flask.request.content_type == 'text/plain':
    #    text = flask.request.data.decode('utf-8')
    if flask.request.content_type == 'application/json':
        query_data = flask.request.data.decode('utf-8')
        data = json.loads(query_data, encoding='utf-8')
        #text = data["text"]
    else:
        return flask.Response(response='This predictor only supports text or json data', status=415, mimetype='application/json')

    print("data received: {}".format(data))
    #input = {'text': ['hello I am ALexis', "how are you", "are you doing fine??"],
    #            'bbox': ["12 34 87 40, 12 34 87 40, 12 34 87 40, 12 34 87 40", "12 34 87 40, 12 34 87 40, 12 34 87 40",
    #                    "12 34 87 40, 12 34 87 40, 12 34 87 40, 12 34 87 40"]}
    # Do the prediction
    output = ScoringService.predict(model_path, data)

    # Return json response with prediction
    result = {"predictions": output}
    response = json.dumps(result)
    return flask.Response(response=response, status=200, mimetype='application/json')
