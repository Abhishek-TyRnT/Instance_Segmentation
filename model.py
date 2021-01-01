import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D,Dense,Dropout,Flatten,Reshape,MaxPool2D
from tensorflow.keras.applications import VGG16
from tensorflow.image import resize

def get_VGG(input_shape):
    model = VGG16(include_top=False, input_shape = input_shape)
    inp   = model.input
    output = model.get_layer('block5_conv3').output
    model  = Model(inputs = inp, outputs = output,name= 'VGG_Backbone')
    return model

def get_segmentation_model(input_shape,output_shape,kernel_regularizer = None):
    input = Input(shape = input_shape)

    network = Conv2D(512, kernel_size= (3,3), padding= 'SAME', activation= 'relu',kernel_regularizer= kernel_regularizer)(input)
    network = Flatten()(network)
    network = Dense(512, kernel_regularizer= kernel_regularizer)(network)
    network = Dense(3136, kernel_regularizer= kernel_regularizer)(network)
    network = Reshape(target_shape= (56,56))(network)
    network = tf.expand_dims(network, -1)
    network = resize(network, size = output_shape)
    network = network[:,:,:,0]
    model   = Model(inputs= input, outputs= network, name = 'Segmentation')
    return model

def get_score_model(input_shape,kernel_regularizer = None):
    input = Input(shape= input_shape)

    network = MaxPool2D(pool_size= (2, 2), strides= 1, padding= 'SAME')(input)
    network = Flatten()(network)
    network = Dense(512,'relu',kernel_regularizer= kernel_regularizer)(network)
    network = Dropout(0.5)(network)
    network = Dense(1024,'relu',kernel_regularizer= kernel_regularizer)(network)
    network = Dropout(0.5)(network)
    network = Dense(1,kernel_regularizer= kernel_regularizer)(network)
    model   = Model(inputs = input, outputs = network, name = 'score')
    return model

class DeepMask(Model):
    def __init__(self,input_shape,output_shape,kernel_regularizer = None,names = 'DeepMask'):
        super(DeepMask,self).__init__(name = names)
        self.vgg = get_VGG(input_shape)
        vgg_output = self.vgg.output_shape[1:]
        self.segmentation_model = get_segmentation_model(vgg_output, output_shape, kernel_regularizer)
        self.score_model        = get_score_model(vgg_output)

    def call(self,inp):

        network = self.vgg(inp)
        segmentation = self.segmentation_model(network)
        score = self.score_model(network)

        return {'mask': segmentation, 'score': score}

