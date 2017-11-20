import os
import time
import shutil
import socket
import subprocess
import numpy as np

from tensorflow_serving_client import TensorflowServingClient

from model_converters import KerasToTensorflow
from ml_tools import load_image, get_model_spec
import keras_model_specs.model_spec as model_spec


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
}


def assert_lists_same_items(list1, list2):
    assert sorted(list1) == sorted(list2)


def test_convert_tests_cover_all_model_types():
    assert_lists_same_items(model_spec.BASE_SPEC_NAMES, MODEL_SERVING_PORTS.keys())


def setup_model(name, model_path):
    tf_model_dir = '.cache/models/%s' % (name, )

    model_spec = get_model_spec(name)
    model = model_spec.klass(weights='imagenet', input_shape=tuple(model_spec.target_size))
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model.save(model_path)

    if os.path.exists(tf_model_dir):
        shutil.rmtree(tf_model_dir)
    tf_model_dir_dir = os.path.dirname(tf_model_dir)
    if not os.path.exists(tf_model_dir_dir):
        os.makedirs(tf_model_dir_dir)

    return tf_model_dir


def restart_serving_container(model_name):
    subprocess.call(['docker-compose', 'restart', model_name])
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


def assert_converted_model(tf_model_dir):
    assert os.path.exists(tf_model_dir)
    assert os.path.exists(tf_model_dir + '/variables')
    assert os.path.exists(tf_model_dir + '/variables/variables.data-00000-of-00001')
    assert os.path.exists(tf_model_dir + '/variables/variables.index')
    assert os.path.exists(tf_model_dir + '/saved_model.pb')


def assert_model_serving(model_name, imagenet_dictionary, expected_top_5):
    model_spec = get_model_spec(model_name)
    client = TensorflowServingClient('localhost', MODEL_SERVING_PORTS[model_name])
    image_data = load_image('tests/fixtures/files/cat.jpg', model_spec.target_size,
                            preprocess_input=model_spec.preprocess_input)
    result = client.make_prediction(image_data, 'image')
    assert 'class_probabilities' in result
    assert len(result['class_probabilities']) == 1
    assert len(result['class_probabilities'][0]) == 1000
    predictions = result['class_probabilities'][0]
    predictions = list(zip(imagenet_dictionary, predictions))
    predictions = sorted(predictions, reverse=True, key=lambda kv: kv[1])[:5]
    predictions = [(label, float(score)) for label, score in predictions]
    print(predictions)
    classes = [name for name, _ in predictions]
    expected_classes = [name for name, _ in expected_top_5]
    assert classes == expected_classes
    scores = [score for _, score in predictions]
    expected_scores = [score for _, score in expected_top_5]
    np.testing.assert_array_almost_equal_nulp(np.array(scores), np.array(expected_scores))


def assert_model_conversion_and_prediction(temp_file, imagenet_dictionary, model_name, expected_top_5):
    tf_model_dir = setup_model(model_name, temp_file)
    KerasToTensorflow.convert(temp_file, tf_model_dir)
    assert_converted_model(tf_model_dir)
    restart_serving_container(model_name)
    assert_model_serving(model_name, imagenet_dictionary, expected_top_5)


def test_convert_imagenet_inception_v3(temp_file, imagenet_dictionary):
    assert_model_conversion_and_prediction(temp_file, imagenet_dictionary, 'inception_v3', [
        ('tiger cat', 0.4716886878013611),
        ('Egyptian cat', 0.127954363822937),
        ('Pembroke, Pembroke Welsh corgi', 0.07338221371173859),
        ('tabby, tabby cat', 0.052391838282346725),
        ('Cardigan, Cardigan Welsh corgi', 0.008323794230818748)
    ])


def test_convert_imagenet_mobilenet(temp_file, imagenet_dictionary):
    assert_model_conversion_and_prediction(temp_file, imagenet_dictionary, 'mobilenet_v1', [
        ('tiger cat', 0.334694504737854),
        ('Egyptian cat', 0.2851393222808838),
        ('tabby, tabby cat', 0.15471667051315308),
        ('kit fox, Vulpes macrotis', 0.03160465136170387),
        ('lynx, catamount', 0.030886519700288773)
    ])


def test_convert_imagenet_resnet50(temp_file, imagenet_dictionary):
    assert_model_conversion_and_prediction(temp_file, imagenet_dictionary, 'resnet50', [
        ('red fox, Vulpes vulpes', 0.3193315863609314),
        ('kit fox, Vulpes macrotis', 0.19359852373600006),
        ('weasel', 0.14291106164455414),
        ('Pembroke, Pembroke Welsh corgi', 0.1395975947380066),
        ('lynx, catamount', 0.04618712514638901)
    ])


def test_convert_imagenet_xception(temp_file, imagenet_dictionary):
    assert_model_conversion_and_prediction(temp_file, imagenet_dictionary, 'xception', [
        ('red fox, Vulpes vulpes', 0.10058529675006866),
        ('weasel', 0.09152575582265854),
        ('Pembroke, Pembroke Welsh corgi', 0.07581676542758942),
        ('tiger cat', 0.0746716633439064),
        ('kit fox, Vulpes macrotis', 0.06751589477062225)
    ])


def test_convert_imagenet_vgg16(temp_file, imagenet_dictionary):
    assert_model_conversion_and_prediction(temp_file, imagenet_dictionary, 'vgg16', [
        ('kit fox, Vulpes macrotis', 0.3090206980705261),
        ('red fox, Vulpes vulpes', 0.21598483622074127),
        ('Egyptian cat', 0.1327403038740158),
        ('tiger cat', 0.11005250364542007),
        ('tabby, tabby cat', 0.08285804092884064)
    ])


def test_convert_imagenet_vgg19(temp_file, imagenet_dictionary):
    assert_model_conversion_and_prediction(temp_file, imagenet_dictionary, 'vgg19', [
        ('red fox, Vulpes vulpes', 0.3812929391860962),
        ('kit fox, Vulpes macrotis', 0.27262774109840393),
        ('tiger cat', 0.08553500473499298),
        ('lynx, catamount', 0.05379556491971016),
        ('Egyptian cat', 0.047869954258203506)
    ])


def test_convert_imagenet_inception_resnet_v2(temp_file, imagenet_dictionary):
    assert_model_conversion_and_prediction(temp_file, imagenet_dictionary, 'inception_resnet_v2', [
        ('red fox, Vulpes vulpes', 0.3812929391860962),
        ('kit fox, Vulpes macrotis', 0.27262774109840393),
        ('tiger cat', 0.08553500473499298),
        ('lynx, catamount', 0.05379556491971016),
        ('Egyptian cat', 0.047869954258203506)
    ])


def test_convert_imagenet_inception_v4(temp_file, imagenet_dictionary):
    assert_model_conversion_and_prediction(temp_file, imagenet_dictionary, 'inception_v4', [
        ('red fox, Vulpes vulpes', 0.3812929391860962),
        ('kit fox, Vulpes macrotis', 0.27262774109840393),
        ('tiger cat', 0.08553500473499298),
        ('lynx, catamount', 0.05379556491971016),
        ('Egyptian cat', 0.047869954258203506)
    ])


def test_convert_imagenet_resnet152(temp_file, imagenet_dictionary):
    assert_model_conversion_and_prediction(temp_file, imagenet_dictionary, 'resnet152', [
        ('red fox, Vulpes vulpes', 0.3812929391860962),
        ('kit fox, Vulpes macrotis', 0.27262774109840393),
        ('tiger cat', 0.08553500473499298),
        ('lynx, catamount', 0.05379556491971016),
        ('Egyptian cat', 0.047869954258203506)
    ])
