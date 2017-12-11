from keras import backend as K
if K.backend()=='tensorflow':
        K.set_image_dim_ordering("th")

class CIFAR10(object):
    def __init__(self):
        from keras.datasets import cifar10
        from keras.utils.np_utils import to_categorical
        
        (train_x, train_y), (test_x, test_y) = cifar10.load_data()
        self.train_x =  train_x.astype('float32')/255
        self.train_y = to_categorical(train_y, 10)
        self.test_x = test_x.astype('float32')/255
        self.test_y = to_categorical(test_y, 10)
