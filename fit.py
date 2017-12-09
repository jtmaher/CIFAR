import pickle

from data import CIFAR10
from model import Model

d = CIFAR10()

for params in (
        (64, 32, 32, 64),
        (64, 32, 32, 96),
        (64, 32, 32, 128),
        (64, 64, 32, 64),
        (64, 64, 32, 96),
        (64, 64, 32, 128),
        (128, 64, 32, 64),
        (128, 64, 32, 96),
        (128, 64, 32, 128),
        (256, 128, 64, 64),
        (256, 128, 64, 96),
        (256, 128, 64, 128)):
    print( 'Fitting model: ', params)
    m = Model(*params, reg=0.)
    m.fit(d, epochs=25, verbose=2)
    m.save('model_' + '_'.join([str(p) for p in params]))
