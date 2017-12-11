from keras import backend as K
if K.backend()=='tensorflow':
        K.set_image_dim_ordering("th")

import numpy as np

class AutoFlow(object):
    def __init__(self, flow):
        self.flow = flow

    def next(self):
        nx = self.flow.next()            
        return (nx, nx)

    def __len__(self):
        return len(self.flow)


from keras import regularizers
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv2D, UpSampling2D
from keras.layers import Activation, Flatten, Dense, Dropout, Reshape

model = Sequential()
drop = 0.1

model.add(Conv2D( 128, (5, 5), padding="same", input_shape=(3,32,32)))
model.add(Activation('relu'))
model.add(Dropout(drop))
model.add(Conv2D( 128, (5, 5), padding="same"))
model.add(Activation('relu'))
model.add(Dropout(drop))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D( 64, (5, 5), padding="same"))
model.add(Activation('relu'))
model.add(Dropout(drop))
model.add(Conv2D( 64, (5, 5), padding="same"))
model.add(Activation('relu'))
model.add(Dropout(drop))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(drop))
model.add(Dense(256, activation='relu'))
model.add(Dropout(drop))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.05,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.05,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images


from data import CIFAR10


batch_size=128
steps_per_epoch=50000/batch_size
epochs=199
data= CIFAR10()
datagen.fit(data.train_x)
flow = datagen.flow(data.train_x,
                    data.train_y,
                    batch_size=batch_size)

         
model.fit_generator(
    flow,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=(data.test_x, data.test_y),
    verbose=1,
    workers=6)
