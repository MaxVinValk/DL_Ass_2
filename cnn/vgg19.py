from keras import backend as K
# TODO These imports will be redundant, if this file is imported, by a file which already uses these import statements
import tensorflow as tf
from tensorflow import keras
from keras.applications.vgg19 import VGG19
from keras import Model
from tensorflow.keras import layers


def create_models(input_shape, loss_layers):
    """Creating sub-models of VGG19 for loss computation per layer

    Args:
        input_shape (int, int, int): input shape of the image
        loss_layers (List[int]): nth layers of which the loss should be computed

    Returns:
        List[tf.keras.Model]: List of the sub-models of VGG19
    """
    # Classes not defined? classes=<number of classes>
    # If there is some kind of 'CERTIFICATE_VERIFY_FAILED' error, go to: 'Appplications/Python 3.6', and double click 'Install Certificates.command'
    model_vgg = VGG19(input_shape=input_shape,
                      weights="imagenet", include_top=False)

    # Extra print statements
    # print('vgg summary:')
    # model_vgg.summary()

    # Building all models
    models = []
    for idx, layer in enumerate(loss_layers):
        # Take specific layers of the VGG19 model
        layers = model_vgg.layers[(
            0 if idx == 0 else loss_layers[idx-1]+1):layer+1]  # '+1', since the Input is seen as a layer
        model = keras.Sequential(name="model_" + str(idx), layers=layers)
        model.trainable = False

        # Setting correct input shape
        input_shape = (input_shape if idx ==
                       0 else models[idx - 1].output_shape)
        model.build(input_shape=(input_shape))

        # Extra print statements
        # for l in layers:
        #     print(l)
        # print('inpshape', input_shape)
        # model.summary()

        models.append(model)
    return models


def calculate_fp_loss(models, img1, img2):
    """Calculating feature perceptual loss of 2 images

    Args:
        models (List[tensorflow.keras.Model]): List of sub-models of VGG19
        img1 (tensorflow.python.data.ops.dataset_ops.MapDataset): first image
        img2 (tensorflow.python.data.ops.dataset_ops.MapDataset): second image

    Returns:
        int: Feature perceptual loss
    """
    losses = []
    for idx, model in enumerate(models):
        prediciton_1 = model.predict(
            img1 if idx == 0 else prediciton_1, batch_size=1)  # returns np array
        prediciton_2 = model.predict(
            img2 if idx == 0 else prediciton_2, batch_size=1)  # returns np array
        # FIXME Using mean squared error (= squared euclidean distance?)
        losses.append(tf.keras.losses.mean_squared_error(
            prediciton_1, prediciton_2))
    print('losses:', losses)
    return sum(losses)

# TODO remove, this is for test purposes only.


def load_celeba(folder, batch_size, image_size):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        folder, image_size=image_size, batch_size=batch_size)

    normalization_layer = layers.experimental.preprocessing.Rescaling(1. / 255)
    normalized_ds = train_ds.map(lambda x, y: normalization_layer(x))
    print(normalized_ds)
    return normalized_ds


# Creating the models
models = create_models(input_shape=(128, 128, 3), loss_layers=[3, 6, 11])
# Getting one image
img1 = load_celeba("celeba_one_image/data", 1, (128, 128))
img2 = load_celeba("celeba_one_image2/data", 1, (128, 128))
print('img type', type(img1))
# img2 = img1  # TODO img2 should be the generated image
# Calculate the loss for a single image
loss = calculate_fp_loss(models, img1, img2)

# print('out', output_values)
# print('outshape', output_values.shape)
# layer_name_to_output_value = dict(zip(output_names, output_values))
# print(layer_name_to_output_value)
