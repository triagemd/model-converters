Tools for converting Keras models for use with other ML frameworks.

[![Build Status](https://travis-ci.org/triagemd/model-converters.svg?branch=master)](https://travis-ci.org/triagemd/model-converters)
[![PyPI version](https://badge.fury.io/py/model-converters.svg)](https://badge.fury.io/py/model-converters)
[![Docker Pulls](https://img.shields.io/docker/pulls/triage/model-converter.svg)](https://hub.docker.com/r/triage/model-converter/)
[![codecov](https://codecov.io/gh/triagemd/model-converters/branch/master/graph/badge.svg)](https://codecov.io/gh/triagemd/model-converters)

# Getting started

As a Python package:
`pip install --upgrade model-converters`

As a Docker image:
```bash
docker run --rm -i -e MODEL_INPUT=<stored-compatible path> \
				   -e MODEL_OUTPUT=<stored-compatible path> \
				   -e PYTHON_VERSION=<2 or 3> \
				   -e KERAS_VERSION=<...> \
				   -e TENSORFLOW_VERSION=<...> \
				   triage/model-converter:latest
```
(you can add mounts if you want to use a local file)


# Testing

 - `script/test`


# Linting

 - `script/autolint`
