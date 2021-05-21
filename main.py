from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import utils
import keras
from model import SP_Sequential
from layers import SP_Dense

from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
Y = iris.target



# set the hyperparameters
population = 10


# prepare datasets
mnist = keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# encode datasets
encoded_X_train = []
encoded_Y_train = []
index = 0
X_train = X_train[:2]
Y_train = Y_train[:2]
for x, y in zip(X_train, Y_train):

    encoded_X_train.append(utils.GRF_encoder(utils.flatten(x), [0, 255], population, 40))
    encoded_Y_train.append(utils.Y_encoder(y, 10, 30, 10))
    index += 1
    print(index)

print("bb")
# encoded_X_train2 = utils.GRF_encoder(utils.flatten(X_train[0]), [0, 255], population, 40)
#
# gab = encoded_X_train - encoded_X_train2
# argmax = gab.argmax()
# argmin = gab.argmin()




model = SP_Sequential(n_terminals=5, delay=3, tau=7, theta=3)
# model.add(SP_Dense(population*28*28, weight_initializer='random'))
model.add(SP_Dense(28*28*population, weight_initializer='random'))
model.add(SP_Dense(10, weight_initializer='random'))
model.add(SP_Dense(10, weight_initializer='random'))

model.compile(loss='mse', optimizer='my_optimizer', lr=0.01)

my_x_train = [22, 15, 26, 2, 5, 59, 78, 44, 7, 15, 22, 75]

model.fit(encoded_X_train, encoded_Y_train, epochs=100, batch_size=1)

#mse = model.evaluate(x, y, batch_size=1)
#print('mse : ', mse)

