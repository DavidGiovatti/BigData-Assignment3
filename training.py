import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np
import random
import h5py
# Import data
import csv
shuju = []
biaoqian = []

with open ('train.csv') as f:
	reader = csv.reader(f)
	for row in reader:
		shuju.append(row[1])
		biaoqian.append(row[2])
del shuju[0]
del biaoqian[0]

# Convert letters to integers
label=np.zeros((2000,1))
data=np.zeros((2000,14))
label=np.reshape(biaoqian, (2000,1))

def switch(letter=''):
	if letter=='A':
		return int(10)
	elif letter=='C':
		return int(11)
	elif letter=='G':
		return int(12)
	else :
		return int(13)

for i in range(2000):
	for j in range(14):
		data[i][j] = switch(shuju[i][j])

# Initialize Network
model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(64, activation='relu', input_dim=14))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(2, activation='softmax'))

adamx= keras.optimizers.Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
model.compile(loss='sparse_categorical_crossentropy', optimizer=adamx, metrics=['accuracy'])
model.fit(data, label, epochs=2000, batch_size=400)
score ,acc = model.evaluate(data, label, batch_size=400)
print('Test score:', score)
print('Test accuracy:', acc)
model.save("./model.h5")
