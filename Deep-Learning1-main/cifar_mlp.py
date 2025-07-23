from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.utils import to_categorical
from keras.datasets import cifar10
import matplotlib.pyplot as plt
(x_train,y_train),(x_test,y_test) = cifar10.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
model.add(Flatten(input_shape = (32,32,3)))
model.add(Dense(units = 10, activation= 'softmax'))

model.compile(optimizer = 'adam', loss= 'categorical_crossentropy',metrics=['accuracy'])


model.fit(x_train,y_train, epochs =10, batch_size= 64)

model.evaluate()