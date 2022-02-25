import tensorflow as tf
import numpy as np
import functools

def ResBlock(inputs, dim, ks=3, bn=False, activation='relu', stride=1):
    x = inputs
    if bn:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation)(x)
    x = tf.keras.layers.Conv2D(dim, ks, stride, padding='same')(x)
    if bn:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation)(x)
    x = tf.keras.layers.Conv2D(dim, ks, padding='same')(x)
    if stride > 1:
        inputs = tf.keras.layers.Conv2D(dim, ks, stride, padding='same')(inputs)
    return inputs + x

def make_f(input_shape, level):
    xin = tf.keras.layers.Input(input_shape)
    
    x = tf.keras.layers.Conv2D(64, 3, 1, padding='same')(xin)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(2)(x)    
    x = ResBlock(x, 64)
    
    if level == 1:
        return tf.keras.Model(xin, x)
    x = ResBlock(x, 128, stride=2)
    if level == 2:
        return tf.keras.Model(xin, x)
    x = ResBlock(x, 128)
    if level == 3:
        return tf.keras.Model(xin, x)
    x = ResBlock(x, 256, stride=2)
    if level <= 4:
        return tf.keras.Model(xin, x)    
    else:
        raise Exception('No level %d' % level)

def make_g(input_shape, class_num, level):
    xin = tf.keras.layers.Input(input_shape)
    x = xin
    if level == 1:
        x = ResBlock(x, 128, stride=2)
    if level <= 2:
        x = ResBlock(x, 128)
    if level <= 3:
        x = ResBlock(x, 256, stride=2)

    x = ResBlock(x, 256)
    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.BatchNormalization()(x)
    if class_num == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = class_num

    # x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(units, activation=activation)(x)
    return tf.keras.Model(xin, x)

"""
Note that the attacker does not know the model architecture of the client,
but only knows the output dimension of the client model.
"""
def make_e(input_shape, level):
    xin = tf.keras.layers.Input(input_shape)
    act = "relu"
    x = tf.keras.layers.Conv2D(64, 3, 2, padding='same', activation=act)(xin)
    if level == 1:
        x = tf.keras.layers.Conv2D(64, 3, 1, padding='same')(x)
        return tf.keras.Model(xin, x)
    x = tf.keras.layers.Conv2D(128, 3, 2, padding='same', activation=act)(x)
    if level <= 3:
        x = tf.keras.layers.Conv2D(128, 3, 1, padding='same')(x)
        return tf.keras.Model(xin, x)
    x = tf.keras.layers.Conv2D(256, 3, 2, padding='same', activation=act)(x)
    if level <= 4:
        x = tf.keras.layers.Conv2D(256, 3, 1, padding='same')(x)
        return tf.keras.Model(xin, x)
    else:
        raise Exception('No level %d' % level)

"""
This critic/discriminator architecture follows the one from FSHA.
"""
def make_c(input_shape, level):
    xin = tf.keras.layers.Input(input_shape)
    if level == 1:
        x = tf.keras.layers.Conv2D(128, 3, 2, padding='same', activation='relu')(xin)
        x = tf.keras.layers.Conv2D(256, 3, 2, padding='same')(x)
    if level <= 3:
        x = tf.keras.layers.Conv2D(256, 3, 2, padding='same')(xin)
    if level <= 4:
        x = tf.keras.layers.Conv2D(256, 3, 1, padding='same')(xin)
    bn = False
    x = ResBlock(x, 256, bn=bn)
    x = ResBlock(x, 256, bn=bn)
    x = ResBlock(x, 256, bn=bn)
    x = ResBlock(x, 256, bn=bn)
    x = ResBlock(x, 256, bn=bn)
    x = ResBlock(x, 256, bn=bn)
    x = tf.keras.layers.Conv2D(256, 3, 2, padding='same')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(xin, x)

"""
Note that this is a generator only for 32x32x3 images
"""
def make_d(input_shape, channels=3):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(8*8*256, use_bias=False, input_shape=input_shape))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Reshape((8, 8, 256)))
    assert model.output_shape == (None, 8, 8, 256)  # Note: None is the batch size

    model.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 8, 128)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 64)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(channels, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 32, 32, 3)

    return model

def make_resnet(split, class_num):
    return (
        functools.partial(make_f, level=split),
        functools.partial(make_g, level=split, class_num=class_num),
        functools.partial(make_e, level=split),
        functools.partial(make_d),
        functools.partial(make_c, level=split))