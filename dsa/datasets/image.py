import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow_datasets as tfds
from PIL import Image
import os
import pandas as pd

"""
This file contains a few static methods to load image datasets.
For DSA attack, we need a target dataset and an auxiliary dataset.
These two datasets should have no overlap.
By default, when the two datasets come from the same dataset, we
set the target dataset to be five times large as the aux dataset.
All load methods returns a list of two tf.data.Dataset.
"""

def make_dataset(x, y, f):
    x = tf.data.Dataset.from_tensor_slices(x).map(f)
    y = tf.data.Dataset.from_tensor_slices(y)
    return tf.data.Dataset.zip((x, y)).shuffle(1000)

def load_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    
    def resize_clip(x):
        x = x[:,:,None]
        x = tf.tile(x, (1,1,3))
        x = tf.image.resize(x, (32, 32))
        x = x/(255.0/2.0)-1.0
        x = tf.clip_by_value(x, -1., 1.)
        return x
    
    target_ds = make_dataset(x_train, y_train, resize_clip)
    aux_ds = make_dataset(x_test, y_test, resize_clip)

    return target_ds, aux_ds

def load_cifar10(take_first=-1):
    if take_first == -1:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)
        
        target_ds = make_dataset(x_train, y_train, lambda x: tf.clip_by_value(x/(255.0/2.0)-1.0, -1., 1.))
        aux_ds = make_dataset(x_test, y_test, lambda x: tf.clip_by_value(x/(255.0/2.0)-1.0, -1., 1.))
        
        return target_ds, aux_ds
    else:
        # use the first take_num images of x_train as examples
        (x_train, _), (_, _) = tf.keras.datasets.cifar10.load_data()
        return (x_train[0:take_first,:,:,:] / (255.0/2.0)) - 1.0

def load_cifar100(take_first=-1):
    if take_first == -1:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)
        
        target_ds = make_dataset(x_train, y_train, lambda x: tf.clip_by_value(x/(255.0/2.0)-1.0, -1., 1.))
        aux_ds = make_dataset(x_test, y_test, lambda x: tf.clip_by_value(x/(255.0/2.0)-1.0, -1., 1.))
        
        return target_ds, aux_ds
    else:
        # use the first take_num images of x_train as examples
        (x_train, _), (_, _) = tf.keras.datasets.cifar100.load_data()
        return (x_train[0:take_first,:,:,:] / (255.0/2.0)) - 1.0

def load_cifar100vs10():
    (x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()
    (_, _), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)

    # target is cifar10
    target_ds = make_dataset(x_train, y_train, lambda x: tf.clip_by_value(x/(255.0/2.0)-1.0, -1., 1.))
    # aux is cifar100
    aux_ds = make_dataset(x_test, y_test, lambda x: tf.clip_by_value(x/(255.0/2.0)-1.0, -1., 1.))

    return target_ds, aux_ds

def load_cifar10vs100():
    (x_train, y_train), (_, _) = tf.keras.datasets.cifar100.load_data()
    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)

    # target is cifar100
    target_ds = make_dataset(x_train, y_train, lambda x: tf.clip_by_value(x/(255.0/2.0)-1.0, -1., 1.))
    # aux is cifar10
    aux_ds = make_dataset(x_test, y_test, lambda x: tf.clip_by_value(x/(255.0/2.0)-1.0, -1., 1.))

    return target_ds, aux_ds

def load_cat_vs_dog(take_first=-1):
    ds = tfds.load("cats_vs_dogs")
    x = [x["image"].numpy() for x in ds["train"]]
    y = [x["label"].numpy() for x in ds["train"]]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_train = np.array([tf.image.resize(i, size=(32,32)).numpy() for i in x_train])
    x_test = np.array([tf.image.resize(i, size=(32,32)).numpy() for i in x_test])

    if take_first == -1:
        target_ds = make_dataset(x_train, y_train, lambda x: tf.clip_by_value(x/(255.0/2.0)-1.0, -1., 1.))
        aux_ds = make_dataset(x_test, y_test, lambda x: tf.clip_by_value(x/(255.0/2.0)-1.0, -1., 1.))
        return target_ds, aux_ds
    else:
        return x_train[0:take_first,:,:,:]/(255.0/2.0)-1.0

def load_celeba(image_num=10000, take_first=-1):
    celeba_dir = "/home/z/zhux105/celeba-dataset/img_align_celeba"
    x = []
    for i in range(image_num):
        x.append(np.array(Image.open(os.path.join(celeba_dir, "%06d.jpg" % (i+1))))[20:-20,:,:])
    x = np.array(x)
    df = pd.read_csv("/home/z/zhux105/celeba-dataset/list_attr_celeba.csv")
    y = np.array([int(i==1) for i in df["Male"].to_numpy()[0:image_num]]).reshape((image_num, 1))
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_train = np.array([tf.image.resize(i, size=(32,32)).numpy() for i in x_train])
    x_test = np.array([tf.image.resize(i, size=(32,32)).numpy() for i in x_test])

    if take_first == -1:
        target_ds = make_dataset(x_train, y_train, lambda x: tf.clip_by_value(x/(255.0/2.0)-1.0, -1., 1.))
        aux_ds = make_dataset(x_test, y_test, lambda x: tf.clip_by_value(x/(255.0/2.0)-1.0, -1., 1.))
        return target_ds, aux_ds
    else:
        return x_train[0:take_first,:,:,:]/(255.0/2.0)-1.0
