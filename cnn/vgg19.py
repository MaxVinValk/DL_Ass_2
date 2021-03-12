from keras import backend as K
# TODO These imports will be redundant, if this file is imported, by a file which already uses these import statements
import tensorflow as tf
from tensorflow import keras
from keras.applications.vgg19 import VGG19
from keras import Model
from tensorflow.keras import layers
# from keras.models import Model

def create_model(height, width):
    input_shape = (height, width, 3)
    # Classes not defined? classes=<number of classes>
    # If there is some kind of 'CERTIFICATE_VERIFY_FAILED' error, go to: 'Appplications/Python 3.6', and double click 'Install Certificates.command'
    model = VGG19(input_shape=input_shape, weights="imagenet", include_top=False)
    outputs = [layer.output for layer in model.layers]
    output_names = [l.name for l in model.layers]
    model.outputs = outputs
    # model = Model(inputs=model_vgg.inputs, outputs=outputs)
    # model_vgg.trainable = False
    return model, output_names

def get_loss(model, image):
    print('predicting...')
    print(image)
    prediction = model.predict(image)
    print('prediction:', prediction)
    pass

#TODO remove, this is for test purposes only.
def load_celeba(folder, batch_size, image_size):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        folder, image_size=image_size, batch_size=batch_size)

    normalization_layer = layers.experimental.preprocessing.Rescaling(1. / 255)
    normalized_ds = train_ds.map(lambda x, y: normalization_layer(x))
    print(normalized_ds)
    return normalized_ds


model, output_names = create_model(128, 128)
model.summary()
img = load_celeba("celeba_vsmall/data", 1, (128, 128))
output_values = model.predict(img)
layer_name_to_output_value = dict(zip(output_names, output_values))
print(layer_name_to_output_value)

# get_loss(model, img)

