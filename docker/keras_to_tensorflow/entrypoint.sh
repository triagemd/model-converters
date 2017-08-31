#!/usr/bin/env bash

if [ -z "$KERAS_VERSION" ]; then
  echo "KERAS_VERSION not specified"
  exit 1
fi
if [ -z "$TENSORFLOW_VERSION" ]; then
  # not sure if TF version really matters, allow a default for now
  export TENSORFLOW_VERSION=1.3.0
fi

if [ "$PYTHON_VERSION" = "2" ]; then
  export PIP=pip2
  export PYTHON=python2
elif [ "$PYTHON_VERSION" = "3" ]; then
  export PIP=pip3
  export PYTHON=python3
else
  echo "PYTHON_VERSION must be set to 2 or 3"
  exit 1
fi

$PIP install -f /pip "tensorflow==$TENSORFLOW_VERSION" "keras==$KERAS_VERSION" h5py model-converters
$PYTHON /usr/local/bin/keras_to_tensorflow "/model_input/$MODEL_INPUT_FILENAME" "/model_output/$MODEL_OUTPUT_DIRNAME"
chmod -R 0777 /model_output
