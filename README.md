Tools for converting Keras models for use with other ML frameworks.

[![Build Status](https://travis-ci.org/triagemd/model-converters.svg?branch=master)](https://travis-ci.org/triagemd/model-converters) [![PyPI version](https://badge.fury.io/py/model-converters.svg)](https://badge.fury.io/py/model-converters) [![Docker Pulls](https://img.shields.io/docker/pulls/triage/model-converter.svg)](https://hub.docker.com/r/triage/model-converter/)

# Getting started

As a Python package:
`pip install --upgrade model-converters`

As a Docker image:
`docker run --rm -i -v "<path to the folder with the model>:/model_input" -e MODEL_INPUT_FILENAME=<...> -v "<path to output location>:/model_output" -e MODEL_OUTPUT_DIRNAME=<...> -e PYTHON_VERSION=<2 or 3> -e KERAS_VERSION=<...> -e TENSORFLOW_VERSION=<...> triage/model-converter:latest`


# Testing

 - `script/test`


# Linting

 - `script/autolint`
