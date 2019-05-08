# -*- coding: utf-8 -*-


from keras.datasets import mnist
import keras
from keras.models import Model
from keras.utils import np_utils
import numpy as np
import Network
from keras import layers
import tensorflow as tf
###use_gpu(if you dont have gpu,ignore it)###
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

###classes###
n_classes=10

(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train=x_train.reshape(x_train.shape[0],28,28,1)
x_test=x_test.reshape(x_test.shape[0],28,28,1)
y_train=np_utils.to_categorical(y_train,n_classes)
y_test=np_utils.to_categorical(y_test,n_classes)
x_train=x_train.astype('float32')
x_test=x_test.astype('float32')
x_train /=255
x_test /=255

inputs=layers.Input(shape=(28,28,1))
x=Network.dv_v_net(inputs)
x=Network.dv_d_net(x)
x=Network.dv_d_net(x)
x=layers.Flatten()(x)
x=layers.Dense(256, activation='relu')(x)
x=layers.Dense(10, activation='sigmoid')(x)

model=Model(inputs=inputs,outputs=x)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(x_train,y_train, nb_epoch=10, batch_size=64, validation_data=(x_test, y_test), shuffle=True)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
