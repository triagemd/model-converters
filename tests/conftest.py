import pytest
import csv
import requests

from tempfile import NamedTemporaryFile
from backports.tempfile import TemporaryDirectory


@pytest.fixture(scope='function')
def temp_dir():
    with TemporaryDirectory() as d:
        yield d


@pytest.fixture(scope='function')
def temp_file():
    with NamedTemporaryFile() as f:
        yield f.name


@pytest.fixture(scope='session')
def imagenet_dictionary():
    response = requests.get('https://s3.amazonaws.com/tf-models-839c7ddd-9cab-49fa-9b42-bde1a842086e/dictionary.csv')
    reader = csv.reader(response.text.splitlines())
    items = sorted(list(reader), key=lambda kv: int(kv[0]))
    return [name for label, name in items]
