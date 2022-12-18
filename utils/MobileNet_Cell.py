from keras.layers import BatchNormalization, DepthwiseConv2D, ReLU, Conv2D

def mobilenet_block(x, f, s=1, alpha = 1):
    """
    construct a mobilenet block which includes DepthwiseConv 2D, Batch Normalization layer, and ReLU to the model
    :x : shape of the input (height, width, channels)
    :alpha : shrinking parameter
    :f : number of filters, the number of channels after this layer will be int(alpha*n_filters)
    :s : stride of the convolution
    return the mobilenet block with multiple layers
    """
    x = DepthwiseConv2D(3, strides=s, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(int(f*alpha), 1, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def Conv_block(x, f, s=1):
    """
    construct a mobilenet block which includes DepthwiseConv 2D, Batch Normalization layer, and ReLU to the model
    :x : shape of the input (height, width, channels)
    :alpha : shrinking parameter
    :f : number of filters, the number of channels after this layer will be int(alpha*n_filters)
    :s : stride of the convolution
    return the convolution block with multiple layers
    """
    x = Conv2D(3, strides=s, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(f, 1, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x