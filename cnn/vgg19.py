from keras import backend as K
# TODO These imports will be redundant, if this file is imported, by a file which already uses these import statements
import tensorflow as tf
from tensorflow import keras
from keras.applications.vgg19 import VGG19
from keras import Model
from tensorflow.keras import layers
# from keras.models import Model


def create_models(height, width, loss_layers):
    input_shape = (height, width, 3)
    # Classes not defined? classes=<number of classes>
    # If there is some kind of 'CERTIFICATE_VERIFY_FAILED' error, go to: 'Appplications/Python 3.6', and double click 'Install Certificates.command'

    model_vgg = VGG19(input_shape=input_shape,
                      weights="imagenet", include_top=False)
    print('vgg summary:')
    model_vgg.summary()
    models = []
    for idx, layer in enumerate(loss_layers):
        model = keras.Sequential(name="model" + str(idx))
        layers = model_vgg.layers[(0 if idx == 0 else loss_layers[idx-1]):layer+1]
        # output_names = [lyr.name for lyr in model.layers]
        # model = keras.Sequential(layers=layers)
        model.trainable = False
        for l in layers:
            print(l)
            model.add(l)
        model.build(input_shape= (input_shape if idx == 0 else models[idx-1].output_shape))
        # model.build(input_shape=model.input_shape)
        model.summary()
        models.append(model)
    return models


def get_loss(model, image):
    print('predicting...')
    print(image)
    prediction = model.predict(image)
    print('prediction:', prediction)
    pass

# TODO remove, this is for test purposes only.


def load_celeba(folder, batch_size, image_size):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        folder, image_size=image_size, batch_size=batch_size)

    normalization_layer = layers.experimental.preprocessing.Rescaling(1. / 255)
    normalized_ds = train_ds.map(lambda x, y: normalization_layer(x))
    print(normalized_ds)
    return normalized_ds


models = create_models(128, 128, loss_layers=[3, 6, 9])

img = load_celeba("celeba_vsmall/data", 1, (128, 128))
output_values = models[0].predict(img)
# layer_name_to_output_value = dict(zip(output_names, output_values))
# print(layer_name_to_output_value)

# get_loss(model, img)
