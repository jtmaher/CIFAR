from keras import backend as K
if K.backend()=='tensorflow':
        K.set_image_dim_ordering("th")

import numpy as np


class Model(object):
    def __init__(self, n1, n2, n3, nh, reg):
        self.reg = reg
        self.sizes = [n1, n2, n3, nh]
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
        
        reg = self.reg
        r=lambda: regularizers.l2(reg)
        (n1, n2, n3, nh) = self.sizes
        model = Sequential()
        def L(n, in_shape=None):
            if in_shape is not None:
                model.add(Conv2D(n, (3,3), padding="same", input_shape=in_shape, 
                                 kernel_regularizer=r(), bias_regularizer=r()))
            else:
                model.add(Conv2D(n, (3,3), padding="same", 
                                 kernel_regularizer=r(), bias_regularizer=r()))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2,2)))

        def M(n):
            model.add(Conv2D(n, (3,3), padding="same", 
                             kernel_regularizer=r(), bias_regularizer=r()))
            model.add(Activation('relu'))
            model.add(UpSampling2D((2,2)))

        L(n1, in_shape=(3,32,32))
        L(n2)
        L(n3)
        model.add(Flatten())
        model.add(Dense(nh, kernel_regularizer=r(), bias_regularizer=r()))
        # activations???
        model.add(Activation('relu'))
        model.add(Dense(n3*4*4, kernel_regularizer=r(), bias_regularizer=r()))
        model.add(Reshape((n3,4,4)))
        M(n3)
        M(n2)
        M(n1)
        model.add(Conv2D(3, (3, 3), padding="same"))
        model.add(Activation('relu'))
        model.compile(optimizer='adam', loss='mean_squared_error')
        ll = model.layers
        
        self.model = model
        self.eval_hidden = K.function([ll[0].input], [ll[11].output])
        
    def fit(self, data, epochs=1, verbose=1, batch_size=128):
         self.model.fit(data.train_x, data.train_x, batch_size=batch_size,
                        validation_data=(data.test_x, data.test_x), 
                        epochs=epochs, verbose=verbose)
            
    def comp(self, i,  feat):
        from matplotlib import pyplot as plt
        plt.figure()
        plt.imshow(np.clip(np.transpose(feat[i], (1,2,0)),0,1))
        plt.figure()
        plt.imshow(np.clip(np.transpose(self.model.predict(feat[i:(i+1),:])[0], (1,2,0)), 0, 1))

    
