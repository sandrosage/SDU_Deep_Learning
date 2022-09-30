from keras.utils import to_categorical
from json import dump, loads
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import preprocessing
from keras.datasets import imdb, mnist
from keras.layers import Dense, Embedding, LSTM, Input, Concatenate, Conv2D, MaxPooling2D, Flatten
from keras.models import Model
from keras.callbacks import ModelCheckpoint

# Exercise 1
# Loads the data
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
# one-hot encoding the labels to work with the output layer
num_classes = 10
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)

inputs = Input((28, 28, 1))
x1 = Conv2D(128, (3, 3), activation='relu')(inputs)
x1 = MaxPooling2D((2, 2))(x1)
x2 = Conv2D(128, (3, 3), activation="relu")(inputs)
x2 = MaxPooling2D((2, 2))(x2)
x = Concatenate()([x1, x2])
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
output = Dense(10, activation='softmax')(x)
model = Model(inputs, output)
model.summary()
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, train_labels, epochs=5, batch_size=64, validation_data=(test_data, test_labels))

# Exercise 2
def history_plot(history):
    epochs = range(1, len(history['acc']) + 1)
    plt.plot(epochs, history['acc'], '--', label='Training accuracy', color='b')
    plt.plot(epochs, history['val_acc'], '-', label='Validation accuracy', color='b')
    plt.plot(epochs, history['loss'], '--', label='Training accuracy', color='r')
    plt.plot(epochs, history['val_loss'], '-', label='Validation accuracy', color='r')

    plt.title('Training and validation accuracy')
    plt.legend()
    plt.show()

max_len = 500
batch_size = 1
NUM_WORDS = 10000
(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.imdb.load_data(num_words=NUM_WORDS)

from keras import preprocessing
train_data = preprocessing.sequence.pad_sequences(train_data, max_len)
test_data = preprocessing.sequence.pad_sequences(test_data, max_len)

inputs = Input(shape=(max_len,))
input_embedding = Embedding(NUM_WORDS, 8, input_length=max_len)(inputs)
x = LSTM(64)(input_embedding)
output = Dense(1, activation="sigmoid")(x)
model = Model(inputs, output)
model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(train_data, train_labels, epochs=5, batch_size=batch_size, validation_data=(test_data, test_labels),
                    callbacks=[ModelCheckpoint(f'Day 7/RNN.h5', monitor='val_accuracy', save_best_only=True)])
with open('Day 7/RNN.txt', 'w') as file:
    dump(history.history, file)
history = loads(open('Day 7/RNN.txt').read())
history_plot(history)

