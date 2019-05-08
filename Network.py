import keras
import numpy as np
from keras import layers

def dv_v_net(x_input):
    conv1_1=layers.Conv2D(16, (3,  3), activation='relu', padding='same', strides=(1, 1))(x_input)
    conv1_2=layers.Conv2D(16, (3,  3), activation='relu', padding='same', strides=(1, 1))(conv1_1)
    maxpool_1=layers.MaxPooling2D(pool_size=(2, 2))(conv1_2)
    x1=layers.ZeroPadding2D((7,7))(maxpool_1)
    conv2_1=layers.Conv2D(16, (3,  3), activation='relu', padding='same', strides=(1, 1))(maxpool_1)
    conv2_2=layers.Conv2D(16, (3,  3), activation='relu', padding='same', strides=(1, 1))(conv2_1)
    maxpool_2=layers.MaxPooling2D(pool_size=(2, 2))(conv2_2)
###keep size
    x=layers.ZeroPadding2D((7,7))(maxpool_2)
    return x
    
def dv_d_net(x):
    x1=layers.Conv2D(16, (3,  3), activation='relu', padding='same', strides=(1, 1))(x)
    x=layers.BatchNormalization()(x)
    x=layers.Activation('relu')(x)
    x2=layers.Conv2D(16, (3,  3), activation='relu', padding='same', strides=(1, 1))(x1)
 
    x3=layers.concatenate([x1, x2] , axis=3)
    x=layers.BatchNormalization()(x3)
    x=layers.Activation('relu')(x)
    x4=layers.Conv2D(32, (3,  3), activation='relu', padding='same', strides=(1, 1))(x)
 
    x5=layers.concatenate([x3, x4] , axis=3)
    x=layers.BatchNormalization()(x5)
    x=layers.Activation('relu')(x)
    x6=layers.Conv2D(64, (3,  3), activation='relu', padding='same', strides=(1, 1))(x)
 
    x7=layers.concatenate([x5, x6] , axis=3)
    x=layers.BatchNormalization()(x7)
    x=layers.Activation('relu')(x)
    x8=layers.Conv2D(124, (3,  3), activation='relu', padding='same', strides=(1, 1))(x)
    
    x=layers.BatchNormalization()(x8)
    x=layers.Activation('relu')(x)
    x9=layers.Conv2D(124, (3,  3), activation='relu', padding='same', strides=(1, 1))(x)
    x9=layers.MaxPooling2D(pool_size=(2, 2))(x9)
    return x9
    
