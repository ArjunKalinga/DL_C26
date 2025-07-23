import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten 
from keras.datasets import cifar10
from keras.utils import to_categorical

# Load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Build model
model = Sequential()
model.add(Flatten(input_shape=(32, 32, 3)))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile    zer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train and save history
history = model.fit(x_train, y_train, epochs=20, batch_size=64)

# Evaluate
model.evaluate(x_test, y_test)

# Plot training loss and accuracy
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['accuracy'], label='Accuracy')
plt.legend()
plt.show() 
plt.title("Train and validation")





