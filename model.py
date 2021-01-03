import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Reshape, MaxPool2D
from tensorflow.keras.applications import VGG16
from tensorflow.image import resize


def get_VGG(input_shape):
    model = VGG16(include_top=False, input_shape=input_shape)
    inp = model.input
    output = model.get_layer('block5_conv3').output
    model = Model(inputs=inp, outputs=output, name='VGG_Backbone')
    return model


def get_segmentation_model(input, output_shape, kernel_regularizer=None):
    network = Conv2D(512, kernel_size=(3, 3), padding='SAME', activation='relu', kernel_regularizer=kernel_regularizer)(
        input)
    network = Flatten()(network)
    network = Dense(512, kernel_regularizer=kernel_regularizer)(network)
    network = Dense(3136, kernel_regularizer=kernel_regularizer)(network)
    network = Reshape(target_shape=(56, 56))(network)
    network = tf.expand_dims(network, -1)
    network = resize(network, size=output_shape)
    network = network[:, :, :, 0]
    return network


def get_score_model(input, kernel_regularizer=None):
    network = MaxPool2D(pool_size=(2, 2), strides=1, padding='SAME')(input)
    network = Flatten()(network)
    network = Dense(512, 'relu', kernel_regularizer=kernel_regularizer)(network)
    network = Dropout(0.5)(network)
    network = Dense(1024, 'relu', kernel_regularizer=kernel_regularizer)(network)
    network = Dropout(0.5)(network)
    network = Dense(1, kernel_regularizer=kernel_regularizer)(network)
    return network


class DeepMask(Model):
    def __init__(self, input_shape, output_shape, kernel_regularizer=None, names='DeepMask'):
        super(DeepMask, self).__init__(name=names)
        vgg = get_VGG(input_shape)
        vgg.trainable = False
        vgg_output = vgg.output
        segmentation_model = get_segmentation_model(vgg_output, output_shape, kernel_regularizer)
        score_model = get_score_model(vgg_output)
        self.model = Model(inputs=[vgg.inputs], outputs=[segmentation_model, score_model])

    def call(self, inp):
        segmentation, score = self.model(inp)

        return {'mask': segmentation, 'score': score}
