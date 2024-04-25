from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt


(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()
num_classes = 10
x_train = X_train.reshape(60000, 784)
x_test = X_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape, 'train input samples')
print(x_test.shape, 'test input samples')
y_train = keras.utils.to_categorical(Y_train, num_classes)
y_test = keras.utils.to_categorical(Y_test, num_classes)
print(y_train.shape, 'train output samples')
print(y_test.shape, 'test output samples')

def bigger_model(epochs=10):
    model = Sequential()
    model.add(layers.Dense(256, activation='sigmoid', input_shape=(784,)))
    model.add(layers.Dense(128, activation='sigmoid'))
    model.add(layers.Dense(10, activation='softmax'))
    model.summary()
    
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs = epochs, validation_data = (x_test, y_test))
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    model_name = "bigger_model"
    
    return model
if __name__=="__main__":
    model=bigger_model(10)
    model.save('MNIST_Model.h5')
    