import os
import time
import shutil
import socket
import subprocess
import numpy as np
import sys

from tempfile import NamedTemporaryFile
from tensorflow_serving_client import TensorflowServingClient
from model_converters import KerasToTensorflow
from ml_tools import load_image
from keras_model_specs import ModelSpec
from keras_model_specs.model_spec import BASE_SPEC_NAMES


MODEL_SERVING_PORTS = {
    'inception_v3': 9001,
    'mobilenet_v1': 9002,
    'resnet50': 9003,
    'xception': 9004,
    'vgg16': 9005,
    'vgg19': 9006,
    'inception_resnet_v2': 9007,
    'inception_v4': 9008,
    'resnet152': 9009,
    'densenet_121': 9010,
    'densenet_169': 9011,
    'densenet_201': 9012,
    'nasnet_large': None,  # takes too long
    'nasnet_mobile': 9013,
}
# no idea why the Python 2 CI won't run these models (they run out of memory)
if sys.version_info[0] == 2:
    MODEL_SERVING_PORTS['inception_v4'] = None
    MODEL_SERVING_PORTS['nasnet_mobile'] = None
    MODEL_SERVING_PORTS['resnet152'] = None
    MODEL_SERVING_PORTS['inception_resnet_v2'] = None
    MODEL_SERVING_PORTS['densenet_169'] = None


def assert_lists_same_items(list1, list2):
    assert sorted(list1) == sorted(list2)


def cat_image(model_spec):
    return load_image('tests/fixtures/files/cat.jpg', model_spec.target_size,
                      preprocess_input=model_spec.preprocess_input)


def setup_model(name, model_path):
    tf_model_dir = '.cache/models/%s' % (name, )

    model_spec = ModelSpec.get(name)
    model = model_spec.klass(weights='imagenet', input_shape=tuple(model_spec.target_size))
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model.save(model_path)

    image_data = cat_image(model_spec)
    expected_scores = model.predict(image_data)

    if os.path.exists(tf_model_dir):
        shutil.rmtree(tf_model_dir)
    tf_model_dir_dir = os.path.dirname(tf_model_dir)
    if not os.path.exists(tf_model_dir_dir):
        os.makedirs(tf_model_dir_dir)

    return tf_model_dir, expected_scores


def start_serving_container(model_name):
    subprocess.call(['docker-compose', 'up', '-d', model_name])
    attempt = 0
    while attempt <= 60:
        attempt += 1
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect(('localhost', MODEL_SERVING_PORTS[model_name]))
            if len(s.recv(1)) > 0:
                break
        except socket.error:
            pass
        time.sleep(1)


def kill_serving_container(model_name):
    subprocess.call(['docker-compose', 'stop', model_name])


def assert_converted_model(tf_model_dir):
    assert os.path.exists(tf_model_dir)
    assert os.path.exists(tf_model_dir + '/variables')
    assert os.path.exists(tf_model_dir + '/variables/variables.data-00000-of-00001')
    assert os.path.exists(tf_model_dir + '/variables/variables.index')
    assert os.path.exists(tf_model_dir + '/saved_model.pb')


def assert_model_serving(model_name, expected_scores):
    model_spec = ModelSpec.get(model_name)
    client = TensorflowServingClient('localhost', MODEL_SERVING_PORTS[model_name])
    result = client.make_prediction(cat_image(model_spec), 'image')

    assert 'class_probabilities' in result
    assert len(result['class_probabilities']) == 1

    scores = result['class_probabilities'][0]
    np.testing.assert_array_almost_equal(np.array(scores), np.array(expected_scores).flatten())


def test_convert_tests_cover_all_model_types():
    assert_lists_same_items(BASE_SPEC_NAMES, MODEL_SERVING_PORTS.keys())


def test_converted_models_have_same_scores():
    for model_name in MODEL_SERVING_PORTS:
        if MODEL_SERVING_PORTS[model_name] is None:
            continue
        print('Testing model ' + model_name)

        with NamedTemporaryFile() as f:
            temp_file = f.name
            tf_model_dir, expected_scores = setup_model(model_name, temp_file)
            KerasToTensorflow.convert(temp_file, tf_model_dir)

            assert_converted_model(tf_model_dir)
            start_serving_container(model_name)
            assert_model_serving(model_name, expected_scores)
            kill_serving_container(model_name)
