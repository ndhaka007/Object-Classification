import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras import backend as K
from keras import optimizers
#import matplotlib.pyplot as plt
import itertools
import h5py

batch_size = 75
epochs = 6


# input image dimensions 96 or 48
size = 128
img_rows, img_cols = size, size

# change it if you need 6 or 7 or 8 (generally)
num_classes = 5

data = np.load('data.npy');

train = data[:3000,:]
test  = data[3000:,:]

X_train = train[:,:9216].reshape(train.shape[0],128,128,1)
X_test = test[:,:9216].reshape(test.shape[0],128,128,1)
Y_train = train[:,9216:].flatten()
Y_test  = test[:,9216:].flatten()


path_data = '/home/eee/ug/14084003/Dhaka/' 
input_shape = (img_rows, img_cols, 1)

x_train = X_train.astype('float32')
x_test = X_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


y_train = keras.utils.to_categorical(Y_train, num_classes)
y_test = keras.utils.to_categorical(Y_test, num_classes)

print y_train.shape , y_test.shape

model = Sequential()
model.add(ZeroPadding2D(padding=(2, 2),  input_shape=input_shape, name='pad1'))
model.add(Conv2D(filters= 64, kernel_size=(5, 5), strides=(1, 1), activation='relu', use_bias=False, name='conv1'))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1'))
model.add(ZeroPadding2D(padding=(2, 2), name='pad2'))
model.add(Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), activation='relu', use_bias=False, name='conv2'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2'))
model.add(ZeroPadding2D(padding=(2, 2), name='pad3'))
model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', use_bias=False, name='conv3'))
model.add(MaxPooling2D(pool_size=(12, 12), strides=(16, 16), name='pool3'))
model.add(Flatten())
model.add(Dense(300, activation='relu', name='fc4'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax', name='prob'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
print model.summary()
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
          verbose=1, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

path = '/home/eee/ug/14084003/Dhaka/'
model.save(path + 'new_weigths.h5')
