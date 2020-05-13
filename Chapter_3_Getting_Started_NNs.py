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
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import imdb, reuters, boston_housing

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
            f' most-used words no word index will exceed 10,000.\nMax Index = '
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

        print(f"There are {len(train_data)} training examples and {len(test_data)} testing examples")

        # Illustration of the input data
        print(
            f'In this dataset the labels denote the topic of the piece. There are 46 topics represented, each one is '
            f'mutually exclusive.\nHaving taken the top 10,000 most-used words no word index will exceed 10,000.\n'
            f'Max Index = {max([max(sequence) for sequence in train_data])}')

        print(
            f"For the sake of illustration, let's decode an article back to English (not being printed for easier reading)")
        word_index = reuters.get_word_index()
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
        x_test = vectorise_sequences(test_data)

        print("For the labels this time around, there are a few options. A very common option is one-hot-encoding, for "
              "which Keras has an in-built function (a manual version is included in the code for educational purposes)")

        def to_one_hot(labels, dimension=46):
            ret = np.zeros((len(labels), dimension))
            for i, label in enumerate(labels):
                ret[i, label] = 1
            return ret

        one_hot_train_labels = to_categorical(train_labels)
        one_hot_test_labels = to_categorical(test_labels)

        # Design and compile the model
        print("Now to build the network, this time using parameters with greater configurability")
        model = models.Sequential()
        model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(46, activation='softmax'))

        model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='categorical_crossentropy',
                      metrics=[metrics.categorical_accuracy])

        # Divide the training data
        print("Creating a validation set for greater insight during training")
        x_val = x_train[:1000]  # Taking the 1st 1000 samples for validation
        partial_x_train = x_train[1000:]  # Leaving everything from 1000 onwards for training
        y_val = one_hot_train_labels[:1000]  # Taking the 1st 1000 labels for validation
        partial_y_train = one_hot_train_labels[1000:]  # Leaving everything from 1000 onwards for training

        # Train the model
        print("Begin training the model:")
        history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))
        history_dict = history.history

        print(f"\nNote that the history returned by the fit function has a 'history' member which is a dictionary. "
              f"The keys are: {history_dict.keys()}")  # ['loss', 'categorical_accuracy', 'val_loss', 'val_categorical_accuracy']

        # Prepare to plot the training and validation information
        loss_values = history_dict['loss']
        val_loss_values = history_dict['val_loss']
        acc_values = history_dict['categorical_accuracy']
        val_acc_values = history_dict['val_categorical_accuracy']

        epochs = range(1, len(history_dict['categorical_accuracy']) + 1)
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
        results = model.evaluate(x_test, one_hot_test_labels)
        print(f"Evaluation Results: Loss = {results[0]}    Accuracy = {results[1] * 100}%")

    def run_boston_housing():
        # Extract useful data from dataset
        print('Extracting the Boston Housing dataset')
        (train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

        print(f"This dataset contains information regarding house prices and stock market values from the mid-1970's."
              f"Let's take a quick look at the data.\ntrain_data.shape = {train_data.shape}\ntest_data.shape = "
              f"{test_data.shape}\nYou'll note immediately that there are fewer data points than in previous examples.")

        print(f"A very big problem we're going to face here is the scaling of different input values. This disparity of "
              f"scaling will lead to a host of problems,\nnot the least of which is divergence. Even if the model did "
              f"converge, the differing scale of each feature does not correspond to predictable shifts in the output.\n"
              f"The best approach when dealing with an issue like this is 'Normalisation', where we subtract the mean of "
              f"the dataset and divide the result by the standard deviation.")

        # Preparing the data
        print(f"    Before Normalisation: \nMean = {train_data.mean(axis=0)}\nstd = {train_data.std(axis=0)}")

        mean = train_data.mean(axis=0)
        train_data -= mean
        std = train_data.std(axis=0)
        train_data /= std

        test_data -= mean
        test_data /= std

        print(f"    After Normalisation: \nMean = {train_data.mean(axis=0)}\nstd = {train_data.std(axis=0)}")
        print("As you see, the result of Normalisation is a dataset where every feature is now 0-centered with unit standard deviation.")

        # In this example we'll need to instantiate the model multiple times. Hence we define a function to do it
        def build_model():
            model = models.Sequential()
            model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
            model.add(layers.Dense(64, activation='relu'))
            model.add(layers.Dense(1))

            model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='mse', metrics=[metrics.mae])
            return model

        print("We mentioned earlier that there are substantially fewer data points in this dataset than we had seen "
              "previously. The problem this presents is a difficulty in validation.\nHow does one prove that the "
              "validation data isn't atypical of the dataset as a whole? To get around this issue a common solution is\n"
              "'K-Fold cross validation, where we instantiate the model k times and train it on k-1 splits of data while "
              "validating on the last split.")

        # Implementing K-fold cross validation
        k = 4
        num_val_samples = len(train_data) // k
        num_epochs = 500
        all_scores = []
        all_mae_histories = []

        for i in range(k):
            print('processing fold #', i)
            val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
            val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

            partial_train_data = np.concatenate(
                [train_data[:i * num_val_samples], train_data[:(i + 1) * num_val_samples]],
                axis=0)
            partial_train_targets = np.concatenate(
                [train_targets[:i * num_val_samples], train_targets[:(i + 1) * num_val_samples]],
                axis=0)

            model = build_model()
            history = model.fit(partial_train_data, partial_train_targets,
                                validation_data=(val_data, val_targets),
                                epochs=num_epochs, batch_size=1, verbose=0)

            mae_history = history.history['val_mean_absolute_error']
            all_mae_histories.append(mae_history)

            test_mse, test_mae = model.evaluate(test_data, test_targets, verbose=0)
            all_scores.append(test_mae)

        print(f"\nKeep in mind during this that due to our normalisation method an mae score of 1.0 means our model was "
              f"incorrect by $1,000. Below is the relevant statistics from testing each rendition of the model."
              f"\nall_scores = {all_scores}\nAverage mae = {np.mean(all_scores)}\nNow We'll plot the average of the "
              f"per-epoch MAE scores")

        # Processing training scores
        average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

        plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
        plt.xlabel('Epochs')
        plt.ylabel('Validation MAE')
        plt.title('Plot of Validation MAE vs. Epoch #')
        plt.show()

        print("To obtain perhaps a more useful plot we'll apply an exponential moving average to the data.")
        plt.clf()

        def smooth_curve(points, factor=0.9):
            smoothed_points = []
            for point in points:
                if smoothed_points:
                    previous = smoothed_points[-1]
                    smoothed_points.append(previous * factor + point * (1 - factor))
                else:
                    smoothed_points.append(point)
            return smoothed_points

        smooth_mae_curve = smooth_curve(average_mae_history)
        plt.plot(range(1, len(smooth_mae_curve) + 1), smooth_mae_curve)
        plt.xlabel('Epochs')
        plt.ylabel('Filtered Validation MAE')
        plt.title('Plot of Filtered Validation MAE vs. Epoch #')
        plt.show()
        plt.clf()

    run_imdb()
    run_reuters()
    run_boston_housing()
