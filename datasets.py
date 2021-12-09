from numpy.core.numeric import identity
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

BUFFER_SIZE = 10000
SIZE = 32
# min_values = None
# max_values = None

getImagesDS = lambda X, n: np.concatenate([x[0].numpy()[None,] for x in X.take(n)])

def parse(x):
    x = x[:,:,None]
    x = tf.tile(x, (1,1,3))
    x = tf.image.resize(x, (SIZE, SIZE))
    x = x / (255/2) - 1
    x = tf.clip_by_value(x, -1., 1.)
    return x

def parseC(x):
    x = x / (255/2) - 1
    x = tf.clip_by_value(x, -1., 1.)
    return x

def make_dataset(X, Y, f):
    x = tf.data.Dataset.from_tensor_slices(X)
    y = tf.data.Dataset.from_tensor_slices(Y)
    x = x.map(f)
    xy = tf.data.Dataset.zip((x, y))
    xy = xy.shuffle(BUFFER_SIZE)
    return xy

def make_dataset_with_property(X,Y,P,f):
    x = tf.data.Dataset.from_tensor_slices(X)
    y = tf.data.Dataset.from_tensor_slices(Y)
    p = tf.data.Dataset.from_tensor_slices(P)
    x = x.map(f)
    xpy = tf.data.Dataset.zip((x, p, y))
    xpy = xpy.shuffle(BUFFER_SIZE)
    return xpy

def load_mnist():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    
    xpriv = make_dataset(x_train, y_train, parse)
    xpub = make_dataset(x_test, y_test, parse)
    
    return xpriv, xpub

# We just assume it's nomalization by feature, for now
def load_credit_card_without_xpub(property_id=None, num2cat=None):
    df = pd.read_excel('./datasets/credit-card.xls', header=1, index_col=0).sample(frac=1)
    x_train = df.drop(columns=["default payment next month"]).to_numpy()
    y_train = df["default payment next month"].to_numpy().reshape((len(x_train), 1)).astype("float32")
    x_test = np.random.rand(*x_train.shape)
    y_test = np.zeros_like(y_train).astype("float32")
    properties = x_train[:,property_id]
    np.random.shuffle(properties)
    x_test[:,property_id] = properties
    if property_id is not None:
        p_train = np.array([num2cat(i) for i in x_train[:,property_id]]).astype("float32")
        p_test = np.array([num2cat(i) for i in x_test[:,property_id]]).astype("float32")

    min_values = df.drop(columns=["default payment next month"]).describe().transpose()['min'].to_numpy()
    max_values = df.drop(columns=["default payment next month"]).describe().transpose()['max'].to_numpy()
    x_train = (x_train-min_values)/(max_values-min_values)
    # x_test has already been normalized, except the property of interest
    x_test[:,property_id] = (x_test[:,property_id] - min_values[property_id])/(max_values[property_id] - min_values[property_id])

    xpriv = make_dataset_with_property(x_train, y_train, p_train, lambda t: t)
    xpub = make_dataset_with_property(x_test, y_test, p_test, lambda t: t)
    return xpriv, xpub

def load_credit_card(normalization=None, property_id=None, num2cat=None):
    df = pd.read_excel('./datasets/credit-card.xls', header=1, index_col=0).sample(frac=1)
    min_values = df.drop(columns=["default payment next month"]).describe().transpose()['min'].to_numpy()
    max_values = df.drop(columns=["default payment next month"]).describe().transpose()['max'].to_numpy()
    x = df.drop(columns=["default payment next month"]).to_numpy()
    if property_id is not None:
        p = np.array([num2cat(i) for i in x[:,property_id]]).astype("float32")
    if normalization == "feature":
        x = (x-min_values)/(max_values-min_values)
    elif normalization == "example":
        x = np.array([i/np.linalg.norm(i) for i in x])
    y = df["default payment next month"].to_numpy().reshape((len(x), 1)).astype("float32")
    if property_id == None:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        xpriv = make_dataset(x_train, y_train, lambda t: t)
        xpub = make_dataset(x_test, y_test, lambda t: t)
    else:
        x_train, x_test, y_train, y_test, p_train, p_test = train_test_split(x, y, p, test_size=0.2, random_state=42)
        xpriv = make_dataset_with_property(x_train, y_train, p_train, lambda t: t)
        xpub = make_dataset_with_property(x_test, y_test, p_test, lambda t: t)
    return xpriv, xpub

# def restore_from_normalized(X):
    # return np.apply_along_axis(lambda x: x * (max_values - min_values) + min_values, 1, X)

def load_mnist_mangled(class_to_remove):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    
    # remove class from Xpub
    (x_test, y_test), _ = remove_class(x_test, y_test, class_to_remove)
    # for evaluation
    (x_train_seen, y_train_seen), (x_removed_examples, y_removed_examples) = remove_class(x_train, y_train, class_to_remove)
    
    xpriv = make_dataset(x_train, y_train, parse)
    xpub = make_dataset(x_test, y_test, parse)
    xremoved_examples = make_dataset(x_removed_examples, y_removed_examples, parse)
    
    xpriv_other = make_dataset(x_train_seen, y_train_seen, parse)
    
    return xpriv, xpub, xremoved_examples, xpriv_other


def load_fashion_mnist():
    mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    
    xpriv = make_dataset(x_train, y_train, parse)
    xpub = make_dataset(x_test, y_test, parse)
    
    return xpriv, xpub

def remove_class(X, Y, ctr):
    mask = Y!=ctr
    XY = X[mask], Y[mask]
    mask = Y==ctr
    XYr = X[mask], Y[mask]
    return XY, XYr

def plot(X, label='', norm=True):
    n = len(X)
    X = (X+1) / 2 
    fig, ax = plt.subplots(1, n, figsize=(n*3,3))
    for i in range(n):
        ax[i].imshow(X[i]);  
        ax[i].set(xticks=[], yticks=[], title=label)
