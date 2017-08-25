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
            time.sleep(5)


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
            print(predictions)
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
        ('tiger cat', 0.4716886878013611),
        ('Egyptian cat', 0.127954363822937),
        ('Pembroke, Pembroke Welsh corgi', 0.07338221371173859),
        ('tabby, tabby cat', 0.052391838282346725),
        ('Cardigan, Cardigan Welsh corgi', 0.008323794230818748)
    ])


def test_convert_imagenet_mobilenet(temp_file, imagenet_dictionary):
    model_name = 'mobilenet_v1'
    tf_model_dir = setup_model(model_name, temp_file)
    KerasToTensorflow.convert(temp_file, tf_model_dir)
    assert_converted_model(tf_model_dir)
    restart_serving_container(model_name)
    assert_model_serving(model_name, imagenet_dictionary, [
        ('tiger cat', 0.334694504737854),
        ('Egyptian cat', 0.2851393222808838),
        ('tabby, tabby cat', 0.15471667051315308),
        ('kit fox, Vulpes macrotis', 0.03160465136170387),
        ('lynx, catamount', 0.030886519700288773)
    ])


def test_convert_imagenet_resnet50(temp_file, imagenet_dictionary):
    model_name = 'resnet50'
    tf_model_dir = setup_model(model_name, temp_file)
    KerasToTensorflow.convert(temp_file, tf_model_dir)
    assert_converted_model(tf_model_dir)
    restart_serving_container(model_name)
    assert_model_serving(model_name, imagenet_dictionary, [
        ('red fox, Vulpes vulpes', 0.3193315863609314),
        ('kit fox, Vulpes macrotis', 0.19359852373600006),
        ('weasel', 0.14291106164455414),
        ('Pembroke, Pembroke Welsh corgi', 0.1395975947380066),
        ('lynx, catamount', 0.04618712514638901)
    ])


def test_convert_imagenet_xception(temp_file, imagenet_dictionary):
    model_name = 'xception'
    tf_model_dir = setup_model(model_name, temp_file)
    KerasToTensorflow.convert(temp_file, tf_model_dir)
    assert_converted_model(tf_model_dir)
    restart_serving_container(model_name)
    assert_model_serving(model_name, imagenet_dictionary, [
        ('red fox, Vulpes vulpes', 0.10058529675006866),
        ('weasel', 0.09152575582265854),
        ('Pembroke, Pembroke Welsh corgi', 0.07581676542758942),
        ('tiger cat', 0.0746716633439064),
        ('kit fox, Vulpes macrotis', 0.06751589477062225)
    ])


def test_convert_imagenet_vgg16(temp_file, imagenet_dictionary):
    model_name = 'vgg16'
    tf_model_dir = setup_model(model_name, temp_file)
    KerasToTensorflow.convert(temp_file, tf_model_dir)
    assert_converted_model(tf_model_dir)
    restart_serving_container(model_name)
    assert_model_serving(model_name, imagenet_dictionary, [
        ('kit fox, Vulpes macrotis', 0.3090206980705261),
        ('red fox, Vulpes vulpes', 0.21598483622074127),
        ('Egyptian cat', 0.1327403038740158),
        ('tiger cat', 0.11005250364542007),
        ('tabby, tabby cat', 0.08285804092884064)
    ])


def test_convert_imagenet_vgg19(temp_file, imagenet_dictionary):
    model_name = 'vgg19'
    tf_model_dir = setup_model(model_name, temp_file)
    KerasToTensorflow.convert(temp_file, tf_model_dir)
    assert_converted_model(tf_model_dir)
    restart_serving_container(model_name)
    assert_model_serving(model_name, imagenet_dictionary, [
        ('red fox, Vulpes vulpes', 0.3812929391860962),
        ('kit fox, Vulpes macrotis', 0.27262774109840393),
        ('tiger cat', 0.08553500473499298),
        ('lynx, catamount', 0.05379556491971016),
        ('Egyptian cat', 0.047869954258203506)
    ])
