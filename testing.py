import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.models import load_model
import numpy as np
import random
import csv
datas = []
labels = []

with open ('test.csv') as f:
	reader = csv.reader(f)
	for row in reader:
		datas.append(row[1])
del datas[0]
# Convert letters to integers
data=np.zeros((400,14))
def switch(letter=''):
	if letter=='A':
		return int(10)
	elif letter=='C':
		return int(11)
	elif letter=='G':
		return int(12)
	else :
		return int(13)

#print (datas[0][0])
for i in range(400):
	for j in range(14):
		data[i][j]=switch(datas[i][j])
model = load_model("./model.h5")
output = model.predict(data,batch_size=100,verbose=0,steps=None)
result = np.zeros((400,2), dtype = int)
for i in range(400):
	result[i][0] = i
	result[i][1] = int(np.argmax(output[i]))

writer=csv.writer(open('result.csv','w'))
title = ['id', 'prediction']
writer.writerow(title)
for row in result:
	writer.writerow(row)
	
