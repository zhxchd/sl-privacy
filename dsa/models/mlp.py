import tensorflow as tf
from tensorflow.keras import layers
from functools import partial

def make_mlp(attr_num, class_num, split, units, ed_act="sigmoid", dropout=0.0):
    assert split >= 1 and split <= 4
    return (
        partial(make_f, units=units, split=split, dropout=dropout),
        partial(make_g, class_num=class_num, units=units, split=split),
        partial(make_e, units=units, act=ed_act),
        partial(make_d, attr_num=attr_num, act=ed_act),
        make_c
    )

def make_mlp_fsha(attr_num, class_num, split, units, ed_act="sigmoid"):
    assert split >= 1 and split <= 4
    return (
        partial(make_f, units=units, split=split),
        partial(make_g, class_num=class_num, units=units, split=split),
        partial(make_e, units=units, act=ed_act),
        partial(make_e_inverse, units=units, attr_num=attr_num, act=ed_act),
        make_c
    )

def make_f(input_shape, units, split, act="relu", dropout=0.0):
    xin = layers.Input(input_shape)
    x = xin
    # client side hidden layers
    for _ in range(split):
        x = layers.Dense(units, activation=act)(x)
    
    x = layers.Dropout(dropout)(x)
    return tf.keras.Model(xin, x)

def make_f_inverse(input_shape, units, split, attr_num, act="relu"):
    xin = layers.Input(input_shape)
    x = xin
    # client side hidden layers
    for _ in range(split-1):
        x = layers.Dense(units, activation=act)(x)
    
    x = layers.Dense(attr_num, activation="sigmoid")(x)
    
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

def make_e(input_shape, units, act):
    xin = layers.Input(input_shape)
    x = layers.Dense(units*6, activation=act)(xin)
    x = layers.Dense(units*3, activation=act)(x)
    x = layers.Dense(units, activation=act)(x)
    return tf.keras.Model(xin, x)

def make_e_inverse(input_shape, units, attr_num, act):
    xin = layers.Input(input_shape)
    x = layers.Dense(units*3, activation=act)(xin)
    x = layers.Dense(units*6, activation=act)(x)
    x = layers.Dense(attr_num, activation="sigmoid")(x)
    return tf.keras.Model(xin, x)

def make_d(input_shape, attr_num, act):
    xin = layers.Input(input_shape)
    x = layers.Dense(attr_num*6, activation=act)(xin)
    x = layers.Dense(attr_num*3, activation=act)(x)
    x = layers.Dense(attr_num, activation="sigmoid")(x)
    return tf.keras.Model(xin, x)

def make_c(input_shape):
    xin = layers.Input(input_shape)
    x = layers.Dense(1024, activation="relu")(xin)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(1)(x)
    return tf.keras.Model(xin, x)