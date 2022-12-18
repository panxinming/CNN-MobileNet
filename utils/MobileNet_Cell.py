from keras.layers import BatchNormalization, DepthwiseConv2D, ReLU, Conv2D

def mobilenet_block(x, f, s=1, alpha = 1):
    """
    construct a mobilenet block (Depthwise Separable convolutions) which includes 
    DepthwiseConv 2D layer and a pointwise convolution layer, both the two layers
    have Batch Normalization layer and ReLU 
    :x : shape of the input (height, width, channels)
    :alpha : shrinking parameter
    :f : number of filters, the number of channels after this layer will be int(alpha*n_filters)
    :s : stride of the convolution
    return the mobilenet block with multiple layers
    """
    x = DepthwiseConv2D(kernel_size=3, strides=s, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters=int(f*alpha), kernel_size=1, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def Conv_block(x, f, s=1):
    """
    construct a standard convolutional layer with Batch Normalization and ReLU
    :x : shape of the input (height, width, channels)
    :f : number of filters
    :s : stride of the convolution
    return the convolution block with multiple layers
    """
    x = Conv2D(filters=f, kernel_size=3, strides=s, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    return x