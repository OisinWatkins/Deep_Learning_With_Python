"""
Chapter 3: Getting Started With Neural Networks.

This chapter covers:
    -> Core components of neural networks
    -> An introduction to Keras
    -> Setting up a deep learning workstation
    -> Using Neural Networks to solve basic classification and regression problems
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers, optimizers, metrics
from tensorflow.keras.datasets import imdb, reuters

if __name__ == '__main__':
    """
    Run using `python -m Chapter_3_Getting_Started_NNs`
    """

    def run_imdb():
        # Extract useful data from dataset
        print('Extracting the IMDB dataset')
        (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

        # Illustration of the input data
        print(
            f'In this dataset a label of 1 indicates a positive review, 0 a negative review.\nHaving taken the top 10,000'
            f' most-used words no word index will exceed 10,000\nMax Index = '
            f'{max([max(sequence) for sequence in train_data])}')

        print(
            f"For the sake of illustration, let's decode a review back to English (not being printed for easier reading)")
        word_index = imdb.get_word_index()
        reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
        decoded_review = ''.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
        # print(decoded_review)

        # Encoding the inputs
        print("In order to pass these lists of integers into a neural network we must first encode them as tensors of "
              "uniform length.\nIn this example we'll use one-hot encoding, done manually for the sake of understanding.")

        def vectorise_sequences(sequences, dimension=10000):
            ret = np.zeros((len(sequences), dimension))
            for i, sequence in enumerate(sequences):
                ret[i, sequence] = 1
                if i < 1:
                    print(f"\n{sequence} => {ret[i]}\n")
            return ret

        x_train = vectorise_sequences(train_data)
        y_train = np.asarray(train_labels).astype('float32')
        x_test = vectorise_sequences(test_data)
        y_test = np.asarray(test_labels).astype('float32')

        # Design and compile the model
        print("Now to build the network, this time using parameters with greater configurability")
        model = models.Sequential()
        model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='binary_crossentropy',
                      metrics=[metrics.binary_accuracy])

        # Divide the training data
        print("Creating a validation set for greater insight during training")
        x_val = x_train[:10000]  # Taking the 1st 10000 samples for validation
        partial_x_train = x_train[10000:]  # Leaving everything from 10000 onwards for training
        y_val = y_train[:10000]  # Taking the 1st 10000 labels for validation
        partial_y_train = y_train[10000:]  # Leaving everything from 10000 onwards for training

        # Train the model
        print("Begin training the model:")
        history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))
        history_dict = history.history

        print(f"\nNote that the history returned by the fit function has a 'history' member which is a dictionary. "
              f"The keys are: {history_dict.keys()}")  # ['loss', 'binary_accuracy', 'val_loss', 'val_binary_accuracy']

        # Prepare to plot the training and validation information
        loss_values = history_dict['loss']
        val_loss_values = history_dict['val_loss']
        acc_values = history_dict['binary_accuracy']
        val_acc_values = history_dict['val_binary_accuracy']

        epochs = range(1, len(history_dict['binary_accuracy']) + 1)
        plt.plot(epochs, loss_values, 'bo', label='Training Loss')
        plt.plot(epochs, val_loss_values, 'b', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        plt.clf()
        plt.plot(epochs, acc_values, 'bo', label='Training Accuracy')
        plt.plot(epochs, val_acc_values, 'b', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

        # Evaluate the model
        print("\nAfter reviewing each plot, evaluate the performance of the model on new data")
        results = model.evaluate(x_test, y_test)
        print(f"Evaluation Results: Loss = {results[0]}    Accuracy = {results[1] * 100}%")

    def run_reuters():
        # Extract useful data from dataset
        print('Extracting the Reuters dataset')
        (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
