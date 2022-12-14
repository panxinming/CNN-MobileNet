from keras.layers import BatchNormalization, DepthwiseConv2D, ReLU, Conv2D

def mobilenet_block(x, f, s=1, alpha = 1):
    x = DepthwiseConv2D(int(alpha * 3), strides=s, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(f, 1, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def CNN_block(x, f, s=1, alpha = 1):
    x = Conv2D(int(alpha * 3), strides=s, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(f, 1, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x