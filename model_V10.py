# -*- coding:utf-8 -*-
from keras.backend import tensorflow_backend as K
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session(tf.compat.v1.Session(config=config))

l2 = tf.keras.regularizers.l2

def GEI_model_V10(input_shape=(128, 88, 1), num_classes=86, weight_decay=0.00001):

    h = inputs = tf.keras.Input(input_shape)

    h = tf.keras.layers.ZeroPadding2D((3,3))(h)
    h = tf.keras.layers.Conv2D(filters=32,
                               kernel_size=7,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)

    h = tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=2, padding="same")(h)

    h = tf.keras.layers.ZeroPadding2D((3,3))(h)
    h = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=7,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)

    h = tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=2, padding="same")(h)

    h = tf.keras.layers.ZeroPadding2D((1,1))(h)
    h = tf.keras.layers.DepthwiseConv2D(kernel_size=3,
                                        strides=1,
                                        padding="valid",
                                        use_bias=False,
                                        depthwise_regularizer=l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)

    h = tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=2, padding="same")(h)

    h = tf.keras.layers.ZeroPadding2D((1,1))(h)
    h = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=3,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)

    h = tf.keras.layers.Conv2D(filters=128,
                               kernel_size=1,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)

    h = tf.keras.layers.ZeroPadding2D((1,1))(h)
    h = tf.keras.layers.Conv2D(filters=128,
                               kernel_size=3,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)
    
    h = tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=2, padding="same")(h)    

    h = tf.keras.layers.ZeroPadding2D((1,1))(h)
    h = tf.keras.layers.DepthwiseConv2D(kernel_size=3,
                                        strides=1,
                                        padding="valid",
                                        use_bias=False,
                                        depthwise_regularizer=l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)

    h = tf.keras.layers.Conv2D(filters=256,
                               kernel_size=1,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)

    h = tf.keras.layers.GlobalMaxPool2D()(h)

    h = tf.keras.layers.Dense(1024)(h)
    h = tf.keras.layers.Dropout(0.5)(h)

    h1 = tf.keras.layers.Dense(num_classes)(h)

    h = tf.keras.layers.Dense(90)(h)

    h = tf.keras.layers.Reshape((9, 10))(h)

    return tf.keras.Model(inputs=inputs, outputs=[h,h1])

