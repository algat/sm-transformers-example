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
    """Do an inference on a sample data
    Formats allowed are application/json text and application/jsonlines.
    Input
        - application/json: 
                {"data":[{'text': 'hello I am Alexis'}, 
                        {'text': "how are you"},
                        {'text': "are you doing fine??"}]}
        - application/jsonlines: 
                {'text': 'hello I am Alexis'}\n
                {'text': "how are you"}\n
                {'text': "are you doing fine??"}
    Output
        - application/json:
                {"predictions":[{'pred': 'Class1',"proba":0.3},
                            {'pred': 'Class2',"proba":0.6}]}
        - application/jsonlines:
                {'pred': 'Class1',"proba":0.3}\n
                {'pred': 'Class2',"proba":0.6}

    """
    data = None
    if flask.request.content_type == 'application/json':
        query_data = flask.request.data.decode('utf-8')
        query_dict = json.loads(query_data, encoding='utf-8')
        data = query_dict["data"]
    elif flask.request.content_type == 'application/jsonlines':
        query_data = flask.request.data.decode('utf-8')
        data = [json.loads(jline) for jline in query_data.splitlines()]
    else:
        return flask.Response(response='This predictor only supports json data', status=415, mimetype='application/json')

    # convert to dict of lists to list of dicts
    dict_of_lists = {k: [dic[k] for dic in data] for k in data[0]}

    result = ScoringService.predict(model_path, dict_of_lists)

    # convert to list of dicts to dict of lists
    list_of_dict = [dict(zip(result,t)) for t in zip(*result.values())]

    # Send response
    if flask.request.accept_mimetypes['application/json']: # by default it returns json
        response = json.dumps({"predictions": list_of_dict})
        return flask.Response(response=response, status=200, mimetype='application/json')
    elif flask.request.accept_mimetypes['application/jsonlines']:
        response = "\n".join([json.dumps(l) for l in list_of_dict])
        return flask.Response(response=response, status=200, mimetype="application/jsonlines")
    else:
        return flask.Response(response='Accept mimetype not supported', status=415, mimetype='application/json')
