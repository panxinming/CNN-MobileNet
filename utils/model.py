# import packages we need
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import BatchNormalization, Dropout, Input, GlobalAvgPool2D, Dense, Conv2D
from keras import backend as K

from utils.MobileNet_Cell import *



# implement mobile net from scratch
def mobilenet(input_shape, n_classes):  
    input_1 = Input(input_shape)

    x = Conv2D(32, 3, strides=2, padding='same')(input_1)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = mobilenet_block(x, 64)
    x = mobilenet_block(x, 128, 2)
    x = mobilenet_block(x, 128)

    x = mobilenet_block(x, 256, 2)
    x = mobilenet_block(x, 256)

    x = mobilenet_block(x, 512, 2)
    for _ in range(5):
        x = mobilenet_block(x, 512)

    x = mobilenet_block(x, 1024, 2)
    x = mobilenet_block(x, 1024)
  
    x = GlobalAvgPool2D()(x)
  
    output = Dense(n_classes, activation='softmax')(x)
  
    model = Model(input_1, output)
    return model



# Normal CNN

def CNN(input_shape, n_classes):  
    input_1 = Input(input_shape)

    x = Conv2D(32, 3, strides=2, padding='same')(input_1)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv_block(x, 64)
    x = Conv_block(x, 128, 2)
    x = Conv_block(x, 128)

    x = Conv_block(x, 256, 2)
    x = Conv_block(x, 256)

    x = Conv_block(x, 512, 2)
    for _ in range(5):
        x = Conv_block(x, 512)

    x = Conv_block(x, 1024, 2)
    x = Conv_block(x, 1024)
  
    x = GlobalAvgPool2D()(x)
  
    output = Dense(n_classes, activation='softmax')(x)
  
    model = Model(input_1, output)
    return model