# NOTE!: this code is mostly broken and is a work in progress

import coremltools.converters.keras
import keras
import sys


if len(sys.argv) != 2 or not (sys.argv[1].endswith('.h5') or sys.argv[1].endswith('.hdf5')):
    print('usage: keras_to_coreml.py <model file>')
    sys.exit(1)


model_filename = sys.argv[1]
model = keras.models.load_model(model_filename, custom_objects={
    # for mobilenet import, doesn't affect other model types
    'relu6': keras.applications.mobilenet.relu6,
    'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D
})

# cut out to_multi_gpu stuff (this might break some models which don't use to_multi_gpu, just comment it out if so)
stripped_model = next((l for l in model.layers if isinstance(l, keras.engine.training.Model)), None)
if stripped_model:
    model_filename = model_filename + '-stripped.hdf5'
    stripped_model.save(model_filename)
    model = stripped_model


# coreml doesn't support Keras SeparableConv2D layers, so we just treat them as Conv2D layers instead
# this will not work. instead, we will need to write a SeparableConv2D implementation in terms of Conv2D (here's TensorFlow's version: https://github.com/tensorflow/tensorflow/commit/975094c8382ee15dad434278804cb096f9bac84f)
# code below uses private APIs, it will probably break (and doesn't work) -- hopefully coreml ships official support soon

# import coremltools.converters.keras._keras2_converter, coremltools.converters.keras._layers2
# coremltools.converters.keras._keras2_converter._KERAS_LAYER_REGISTRY[keras.layers.convolutional.SeparableConv2D] = coremltools.converters.keras._layers2.convert_convolution


coreml_model = coremltools.converters.keras.convert(
    model, input_names=['image'], image_input_names=['image'], output_names=['class_prob'],
    class_labels=['benign', 'malignant'], predicted_feature_name='lesion_type'
)

# model metadata
coreml_model.author = 'Triage'
coreml_model.short_description = 'Predicts benign vs. malignant lesions.'

# feature descriptions
coreml_model.input_description['image'] = 'Image'
coreml_model.output_description['class_prob'] = 'Probabilities of, in order, [benign, malignant]'

# save
coreml_model.save('BenignMalignant.mlmodel')
