import tensorflow as tf
from tensorflow.keras import layers
from functools import partial

def make_mlp(attr_num, class_num, split, fg_units, fg_act, generator_units):
    assert split >= 1 and split <= 4
    return (
        partial(make_f, units=fg_units, split=split, act=fg_act),
        partial(make_g, class_num=class_num, units=fg_units, split=split, act=fg_act),
        partial(make_generator, attr_num=attr_num, units=generator_units)
    )

def make_resnet(attr_num, class_num, split, d_main, d_inter, generator_units):
    assert split >= 1 and split <= 4
    return (
        partial(make_res_f, d_main = d_main, d_inter=d_inter, split=split),
        partial(make_res_g, class_num=class_num, d_main=d_main, d_inter=d_inter, split=split),
        partial(make_generator, attr_num=attr_num, units=generator_units)
    )

def make_f(input_shape, units, split, act="relu"):
    xin = layers.Input(input_shape)
    x = xin
    # client side hidden layers
    for _ in range(split):
        x = layers.Dense(units, activation=act)(x)
    
    return tf.keras.Model(xin, x)

def make_g(input_shape, class_num, units, split, act="relu"):
    xin = layers.Input(input_shape)
    x = xin
    # remaining hidden layers
    for _ in range(4 - split):
        x = layers.Dense(units, activation=act)(x)
    # output layer
    if class_num == 2:
        x = layers.Dense(1, activation="sigmoid")(x)
    else:
        x = layers.Dense(class_num, activation="softmax")(x)
    return tf.keras.Model(xin, x)

def make_generator(input_shape, attr_num, units):
    xin = tf.keras.layers.Input(input_shape)
    x = xin
    act = "relu"
    for unit in units:
        x = tf.keras.layers.Dense(unit, activation=act)(x)
    x = tf.keras.layers.Dense(attr_num, activation="sigmoid")(x)
    return tf.keras.Model(xin, x)

def resblock(xin, d_main, d_inter, skip=True):
#     ResNetBlock(x) = x + Dropout(Linear(d0, Dropout(ReLU(Linear(d_inter, BatchNorm(x))))))
    x = layers.BatchNormalization()(xin)
    x = layers.Dense(d_inter, activation="relu")(x)
    x = layers.Dense(d_main)(x)
    x = xin + x
    return x

# split is the number of resblocks in
def make_res_f(input_shape, d_main, d_inter, split):
    xin = tf.keras.layers.Input(input_shape)
    x = layers.Dense(d_main)(xin)
    for _ in range(split):
        x = resblock(x, d_main=d_main, d_inter=d_inter)
    return tf.keras.Model(xin, x)

def make_res_g(input_shape, class_num, d_main, d_inter, split):
    xin = tf.keras.layers.Input(input_shape)
    x = xin
    for _ in range(4-split):
        x = resblock(x, d_main=d_main, d_inter=d_inter)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    if class_num == 2:
        x = layers.Dense(1, activation="sigmoid")(x)
    else:
        x = layers.Dense(class_num, activation="softmax")(x)
    return tf.keras.Model(xin, x)