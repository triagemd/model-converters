import os
import keras
import tensorflow
from keras_model_specs.models.custom_layers import Scale
try:
    # for newer versions of Keras
    import keras_applications
except ImportError:
    # for older versions of Keras
    import keras.applications as keras_applications


class KerasToTensorflow(object):

    @staticmethod
    def get_feature_layer(model, layer):
        if isinstance(layer, int):
            return model.layers[layer].output
        elif isinstance(layer, str):
            return model.get_layer(layer).output
        else:
            raise ValueError('Layer must be either a str or an int specifying the model layer, you inputed %s'
                             % str(type(layer)))

    @staticmethod
    def load_keras_model(model_path):
        custom_objects = {
            # just in case you have Lambda layers which implicitly 'import tensorflow as tf'
            # (happens to be the case for some of our internal code)
            'tf': tensorflow,
            'os': os,

            # needed for Resnet152 support
            'Scale': Scale
        }
        try:
            # for mobilenet import, doesn't affect other model types
            custom_objects['relu6'] = keras_applications.mobilenet.layers.ReLU(6, name='relu6')
            custom_objects['DepthwiseConv2D'] = keras_applications.mobilenet.DepthwiseConv2D
        except AttributeError:
            pass

        return keras.models.load_model(model_path, custom_objects=custom_objects)

    @staticmethod
    def convert(model_path, output_dir, feature_layer=None, output_stripped_model_path=None):
        # cut out to_multi_gpu stuff (this could possibly break some models which don't use to_multi_gpu)
        model = KerasToTensorflow.load_keras_model(model_path)
        stripped_model = next((l for l in model.layers if isinstance(l, keras.engine.training.Model)), None)
        if stripped_model:
            if output_stripped_model_path is None:
                output_stripped_model_path = '%s-stripped%s' % os.path.splitext(model_path)
            stripped_model.save(output_stripped_model_path)
            model_path = output_stripped_model_path

        keras.backend.clear_session()
        session = tensorflow.Session()
        keras.backend.set_session(session)

        # disable loading of learning nodes
        keras.backend.set_learning_phase(0)

        model = KerasToTensorflow.load_keras_model(model_path)

        builder = tensorflow.saved_model.builder.SavedModelBuilder(output_dir)

        signature_outputs = {
            'class_probabilities': model.output
        }

        if feature_layer is not None:
            signature_outputs.update({'image_features': KerasToTensorflow.get_feature_layer(model, feature_layer)})

        signature = tensorflow.saved_model.signature_def_utils.predict_signature_def(
            inputs={
                'image': model.input
            },
            outputs=signature_outputs
        )

        builder.add_meta_graph_and_variables(
            sess=keras.backend.get_session(),
            tags=[tensorflow.saved_model.tag_constants.SERVING],
            signature_def_map={
                tensorflow.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature
            }
        )
        builder.save()
