import os
import time
import shutil
import socket
import subprocess

from tensorflow_serving_client import TensorflowServingClient
from grpc.framework.interfaces.face.face import AbortionError

from model_converters import KerasToTensorflow
from ml_tools import load_image, get_model_spec


MODEL_SERVING_PORTS = {
    'inception_v3': 9001,
    'mobilenet_v1': 9002,
    'resnet50': 9003,
    'xception': 9004,
    'vgg16': 9005,
    'vgg19': 9006,
}


def setup_model(name, model_path):
    tf_model_dir = '.cache/models/%s' % (name, )

    model_spec = get_model_spec(name)
    model = model_spec.klass(weights='imagenet', input_shape=model_spec.target_size)
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
    for port in MODEL_SERVING_PORTS.values():
        attempt = 0
        while attempt <= 10:
            attempt += 1
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect(('localhost', port))
                if len(s.recv(1)) > 0:
                    break
            except ConnectionRefusedError:
                pass
            time.sleep(2)


def assert_converted_model(tf_model_dir):
    assert os.path.exists(tf_model_dir)
    assert os.path.exists(tf_model_dir + '/variables')
    assert os.path.exists(tf_model_dir + '/variables/variables.data-00000-of-00001')
    assert os.path.exists(tf_model_dir + '/variables/variables.index')
    assert os.path.exists(tf_model_dir + '/saved_model.pb')


def assert_model_serving(model_name, imagenet_dictionary, expected_top_5):
    model_spec = get_model_spec(model_name)
    attempt = 1
    while True:
        try:
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
            assert predictions == expected_top_5
            break
        except AbortionError as e:
            if attempt > 5:
                raise
            time.sleep(1)
            attempt += 1


def test_convert_imagenet_inception_v3(temp_file, imagenet_dictionary):
    model_name = 'inception_v3'
    tf_model_dir = setup_model(model_name, temp_file)
    KerasToTensorflow.convert(temp_file, tf_model_dir)
    assert_converted_model(tf_model_dir)
    restart_serving_container(model_name)
    assert_model_serving(model_name, imagenet_dictionary, [
        ('tiger cat', 0.47168827056884766),
        ('Egyptian cat', 0.1279538869857788),
        ('Pembroke, Pembroke Welsh corgi', 0.07338253408670425),
        ('tabby, tabby cat', 0.052391838282346725),
        ('Cardigan, Cardigan Welsh corgi', 0.008323835209012032)
    ])


def test_convert_imagenet_mobilenet(temp_file, imagenet_dictionary):
    model_name = 'mobilenet_v1'
    tf_model_dir = setup_model(model_name, temp_file)
    KerasToTensorflow.convert(temp_file, tf_model_dir)
    assert_converted_model(tf_model_dir)
    restart_serving_container(model_name)
    assert_model_serving(model_name, imagenet_dictionary, [
        ('tiger cat', 0.334695041179657),
        ('Egyptian cat', 0.28513845801353455),
        ('tabby, tabby cat', 0.1547166407108307),
        ('kit fox, Vulpes macrotis', 0.03160473331809044),
        ('lynx, catamount', 0.030886217951774597)
    ])


def test_convert_imagenet_resnet50(temp_file, imagenet_dictionary):
    model_name = 'resnet50'
    tf_model_dir = setup_model(model_name, temp_file)
    KerasToTensorflow.convert(temp_file, tf_model_dir)
    assert_converted_model(tf_model_dir)
    restart_serving_container(model_name)
    assert_model_serving(model_name, imagenet_dictionary, [
        ('red fox, Vulpes vulpes', 0.3193321228027344),
        ('kit fox, Vulpes macrotis', 0.19359812140464783),
        ('weasel', 0.14291061460971832),
        ('Pembroke, Pembroke Welsh corgi', 0.13959810137748718),
        ('lynx, catamount', 0.0461868941783905)
    ])


def test_convert_imagenet_xception(temp_file, imagenet_dictionary):
    model_name = 'xception'
    tf_model_dir = setup_model(model_name, temp_file)
    KerasToTensorflow.convert(temp_file, tf_model_dir)
    assert_converted_model(tf_model_dir)
    restart_serving_container(model_name)
    assert_model_serving(model_name, imagenet_dictionary, [
        ('red fox, Vulpes vulpes', 0.10058525949716568),
        ('weasel', 0.09152577072381973),
        ('Pembroke, Pembroke Welsh corgi', 0.07581677287817001),
        ('tiger cat', 0.07467170804738998),
        ('kit fox, Vulpes macrotis', 0.06751599907875061)
    ])


def test_convert_imagenet_vgg16(temp_file, imagenet_dictionary):
    model_name = 'vgg16'
    tf_model_dir = setup_model(model_name, temp_file)
    KerasToTensorflow.convert(temp_file, tf_model_dir)
    assert_converted_model(tf_model_dir)
    restart_serving_container(model_name)
    assert_model_serving(model_name, imagenet_dictionary, [
        ('kit fox, Vulpes macrotis', 0.3090210556983948),
        ('red fox, Vulpes vulpes', 0.21598467230796814),
        ('Egyptian cat', 0.13274021446704865),
        ('tiger cat', 0.11005253344774246),
        ('tabby, tabby cat', 0.08285782486200333)
    ])


def test_convert_imagenet_vgg19(temp_file, imagenet_dictionary):
    model_name = 'vgg19'
    tf_model_dir = setup_model(model_name, temp_file)
    KerasToTensorflow.convert(temp_file, tf_model_dir)
    assert_converted_model(tf_model_dir)
    restart_serving_container(model_name)
    assert_model_serving(model_name, imagenet_dictionary, [
        ('red fox, Vulpes vulpes', 0.3812934458255768),
        ('kit fox, Vulpes macrotis', 0.2726273238658905),
        ('tiger cat', 0.0855349525809288),
        ('lynx, catamount', 0.05379558727145195),
        ('Egyptian cat', 0.04786992818117142)
    ])
