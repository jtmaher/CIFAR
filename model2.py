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

class Model(object):
    def __init__(self, n1, n2, nh, fs, reg):
        self.reg = reg
        self.sizes = [n1, n2, nh]
        self.fs = fs
        self.hourglass()

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        from keras.models import load_model
        self.model = load_model(filename)
        
    def hourglass(self):
        from keras import regularizers
        from keras.models import Sequential
        from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv2D, UpSampling2D
        from keras.layers import Activation, Flatten, Dense, Dropout, Reshape

        fs = self.fs
        reg = self.reg
        r=lambda: regularizers.l2(reg) if reg != 0 else None
        (n1, n2, nh) = self.sizes
        model = Sequential()
        def L(n, in_shape=None):
            if in_shape is not None:
                model.add(Conv2D(n, (fs,fs), padding="same", input_shape=in_shape, 
                                 kernel_regularizer=r(), bias_regularizer=r()))
            else:
                model.add(Conv2D(n, (fs,fs), padding="same", 
                                 kernel_regularizer=r(), bias_regularizer=r()))
            model.add(Activation('relu'))
            model.add(Conv2D(n, (fs,fs), padding="same", 
                             kernel_regularizer=r(), bias_regularizer=r()))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2,2)))

        def M(n):
            model.add(Conv2D(n, (fs,fs), padding="same", 
                             kernel_regularizer=r(), bias_regularizer=r()))
            model.add(Activation('relu'))
            model.add(Conv2D(n, (fs,fs), padding="same", 
                            kernel_regularizer=r(), bias_regularizer=r()))
            model.add(Activation('relu'))
            model.add(UpSampling2D((2,2)))

        L(n1, in_shape=(3,32,32))
        L(n2)

        model.add(Flatten())
        model.add(Dense(nh, kernel_regularizer=r(), bias_regularizer=r()))
        # activations???
        model.add(Activation('relu', name='bottleneck'))
        model.add(Dense(n2*8*8, kernel_regularizer=r(), bias_regularizer=r()))
        model.add(Reshape((n2,8,8)))

        M(n2)
        M(n1)
        model.add(Conv2D(3, (3, 3), padding="same"))
        model.add(Activation('relu'))
        import keras
#        opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)
        model.compile(optimizer='adam', loss='mean_squared_error')
        ll = model.layers
        
        self.model = model
        self.eval_hidden = K.function([ll[0].input], [model.get_layer('bottleneck').output])
        
    def fit(self, data, epochs=1, verbose=1, batch_size=128):
         self.model.fit(data.train_x, data.train_x, batch_size=batch_size,
                        validation_data=(data.test_x, data.test_x), 
                        epochs=epochs, verbose=verbose, shuffle=True)

    def fit_augmented(self, data, epochs=1, verbose=1, batch_size=128, steps_per_epoch=128):
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
         datagen.fit(data.train_x)

         flow = datagen.flow(data.train_x,
                             batch_size=batch_size)

         
         self.model.fit_generator(
                 AutoFlow(flow),
                 steps_per_epoch=steps_per_epoch,
                 epochs=epochs,
                 validation_data=(data.test_x, data.test_x),
                 verbose=verbose,
                 workers=6)
         
    def comp(self, i,  feat):
        from matplotlib import pyplot as plt
        plt.figure()
        plt.imshow(np.clip(np.transpose(feat[i], (1,2,0)),0,1))
        plt.figure()
        plt.imshow(np.clip(np.transpose(self.model.predict(feat[i:(i+1),:])[0], (1,2,0)), 0, 1))

    
