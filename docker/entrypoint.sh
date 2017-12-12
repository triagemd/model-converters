#!/usr/bin/env bash
set -e

if [ -z "$KERAS_VERSION" ]; then
  echo "KERAS_VERSION not specified"
  exit 1
fi
if [ -z "$TENSORFLOW_VERSION" ]; then
  # not sure if TF version really matters, allow a default for now
  export TENSORFLOW_VERSION=1.3.0
fi

if [ "$PYTHON_VERSION" = "2" ]; then
  export PYTHON=python2
elif [ "$PYTHON_VERSION" = "3" ]; then
  export PYTHON=python3
else
  echo "PYTHON_VERSION must be set to 2 or 3"
  exit 1
fi

$PYTHON -m pip install --upgrade pip
$PYTHON -m pip install -f /pip "tensorflow==$TENSORFLOW_VERSION" "keras==$KERAS_VERSION" h5py model-converters
mkdir /input && stored sync "$MODEL_INPUT" /input/model.hdf5
$PYTHON /usr/local/bin/keras_to_tensorflow /input/model.hdf5 /output
stored sync /output "$MODEL_OUTPUT"
