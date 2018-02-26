from keras.models import Sequential
from keras.layers import Dense
import numpy

# load dataset
dataset = numpy.loadtxt("diabetes.csv", delimiter=",")

# split into input and ouput
input = dataset[:,0:8]
output = dataset[:,8]

# create model & add layers
model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

# compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# train the model
model.fit(input, output, epochs=600, batch_size=10, verbose=2)

model.save('diabetes.h5')
