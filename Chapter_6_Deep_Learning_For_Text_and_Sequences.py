"""
Chapter 6: Deep Learning for Text and Sequences

This chapter covers:
    -> Preprocessing text data into useful representations
    -> Working with Recurrent Neural Networks
    -> Using 1D convnets for sequence processing
"""

import os
import string
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras import preprocessing
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.text import Tokenizer

if __name__ == '__main__':
    """
    Run using `python -m Chapter_6_Deep_Learning_For_Text_and_Sequences`
    """

    def one_hot_encode_eg():
        """
        Function which goes through manual and pre-built methods of implementing one-hot encoding for for simple
        datasets

        :return: None
        """
        # Initial data, one entry per sample (in this case a sample is a sentence, however it could be an entire
        # document)
        samples = ["The cat sat on the mat.", "The dog ate my homework."]

        print("First and example of word-level tokenising: ")
        # Build index for all tokens in the data
        token_index = {}

        # Tokenise the samples via the split() method. In real applications you'd also strip out punctuations and
        # special characters
        for sample in samples:
            for word in sample.split():
                if word not in token_index:
                    # Assigns a unique index to each word. Note that nothing is attributed to index 0
                    token_index[word] = len(token_index) + 1

        # We'll only consider the 1st max_len words in each sample
        max_len = 10

        # Array to store the results
        results = np.zeros(shape=(len(samples), max_len, max(token_index.values()) + 1))

        for i, sample in enumerate(samples):
            for j, word in list(enumerate(sample.split()))[:max_len]:
                # Extract the index of the word in the token dictionary and assign the correct value in the vector a
                # value of 1
                index = token_index.get(word)
                results[i, j, index] = 1

        # Print information for comparison
        print(f"samples[0]: '{samples[0]}' becomes:\n {results[0, :, :]}\nToken Matrix Size: {results[0, :, :].shape}\n")
        print(f"samples[1]: '{samples[1]}' becomes:\n {results[1, :, :]}\nToken Matrix Size: {results[1, :, :].shape}\n")

        print("Now an example of characters-level tokenising: ")

        # Gather every printable ASCII character and store in a dictionary
        characters = string.printable
        token_index = dict(zip(range(1, len(characters) + 1), characters))

        # only consider the 1st max_len characters in each sample
        max_len = 50

        # A new array to store the results
        results = np.zeros((len(samples), max_len, max(token_index.keys()) + 1))

        for i, sample in enumerate(samples):
            for j, character in enumerate(sample):
                # Extract the index of each character from the dictionary and assign the corresponding value in the
                # token vector the value of 1
                index = token_index.get(character)
                results[i, j, index] = 1

        # Print information for comparison
        print(f"samples[0]: '{samples[0]}' becomes:\n {results[0, :, :]}\nToken Matrix Size: {results[0, :, :].shape}\n")
        print(f"samples[1]: '{samples[1]}' becomes:\n {results[1, :, :]}\nToken Matrix Size: {results[1, :, :].shape}\n")

        print("Now an example using the Keras in-built word-level tokeniser: ")

        # Creat a tokeniser which only considers the top 1000 most common words, then build the word index
        tokeniser = Tokenizer(num_words=1000)
        tokeniser.fit_on_texts(samples)

        # Turn strings into lists of integer indices
        sequences = tokeniser.texts_to_sequences(samples)

        # You could also directly get the one-hot binary representations. Vectorisation modes other than one-hot
        # encoding are supported. Below is a representation of how to recover the word index that was computed
        one_hot_results = tokeniser.texts_to_matrix(samples, mode='binary')
        word_index = tokeniser.word_index
        print(f"Found {len(word_index)} unique tokens\n")

        print("Now an example using a hashing trick, used when handling a number of unique tokens too large to handle "
              "explicitly: ")

        # Stores words as vectors of size 1000. If you have close to 1000 words or more you'll see many hash collisions,
        # which will decrease the accuracy of this encoding method.
        dimensionality = 1000
        max_len = 10

        results = np.zeros((len(samples), max_len, dimensionality))
        for i, sample in enumerate(samples):
            for j, word in list(enumerate(sample.split()))[:max_len]:
                # Hash the word into a random integer index between 0 and 1000
                index = abs(hash(word)) % dimensionality
                results[i, j, index] = 1

        # Print information for comparison
        print(f"samples[0]: '{samples[0]}' becomes:\n{results[0, :, :]}\nToken Matrix Size: {results[0, :, :].shape}\n")
        print(f"samples[1]: '{samples[1]}' becomes:\n{results[1, :, :]}\nToken Matrix Size: {results[1, :, :].shape}\n")

    def word_embedding_eg():
        """
        Function goes through building and implementing a simple word embedding in neural networks using the IMDB
        dataset

        :return: None
        """
        # Sets the max number of features in the embedded space and the max number of words in the review to use
        max_features = 10000
        max_len = 20

        # Extract the data
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

        # Turns the lists of integers into 2D integer tensors of shape (samples, max_len)
        x_train = preprocessing.sequence.pad_sequences(x_train, max_len)
        x_test = preprocessing.sequence.pad_sequences(x_test, max_len)

        # Build a simple model with an embedding layer as its input
        model = models.Sequential()
        model.add(layers.Embedding(10000, 8, input_length=max_len))
        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation='sigmoid'))

        # Compile the model and train. Don't bother saving the model as this is a simple and fast-training example
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
        model.summary()

        history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

    def pre_trained_embedding_with_imdb():
        """
        Similar to the last function, this function demonstrates the use of embeddings with the IMDB dataset, however
        this time we're using a pre-trained embedding layer: the GloVe embedding created at Stanford.

        :return: None
        """
        # Get the directory of the IMDB dataset
        imdb_dir = 'C:\\Datasets\\IMDB\\aclImdb'
        train_dir = os.path.join(imdb_dir, 'train')

        # Prepare lists for the labels and the text inputs from the dataset
        labels = []
        texts = []

        # Extract all the text inputs, negative first then positive. Store the inputs and the labels in order.
        for label_type in ['neg', 'pos']:
            dir_name = os.path.join(train_dir, label_type)
            for fname in os.listdir(dir_name):
                if fname[-4:] == '.txt':
                    f = open(os.path.join(dir_name, fname), encoding="utf8")
                    texts.append(f.read())
                    f.close()
                    if label_type == 'neg':
                        labels.append(0)
                    else:
                        labels.append(1)

        # Settings for the system: cut reviews off after 100 words, train on 200 samples, validate on 10,000 samples,
        # use only the top 10,000 words
        max_len = 100
        training_samples = 200
        validation_samples = 10000
        max_words = 10000

        # Build the tokeniser
        tokeniser = Tokenizer(num_words=max_words)
        tokeniser.fit_on_texts(texts)
        sequences = tokeniser.texts_to_sequences(texts)

        word_index = tokeniser.word_index
        print(f"Found {len(word_index)} unique tokens\n")

        data = preprocessing.sequence.pad_sequences(sequences, maxlen=max_len)

        labels = np.asarray(labels)
        print(f"Shape of data tensor: {data.shape}")
        print(f"Shape of labels tensor: {labels.shape}\n")

        # Shuffle the data and split it into training and validation sets
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        data = data[indices]
        labels = labels[indices]

        x_train = data[:training_samples]
        y_train = labels[:training_samples]
        x_val = data[training_samples:training_samples + validation_samples]
        y_val = labels[training_samples:training_samples + validation_samples]

        # Open the 100d glove embedding file and extract the embeddings into a dictionary
        glove_dir = 'C:\\Datasets\\GloVe Embeddings'

        embeddings_idx = {}
        f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'), encoding="utf8")
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_idx[word] = coefs
        f.close()
        print(f"Found {len(embeddings_idx)} word vectors\n")

        # Now we need an Embedding matrix which must be of shape (max_words, embedding_dim), where each i contains the
        # embedding_dim dimensional vector for the word of index i in the reference word index. Note that index 0 is
        # only a placeholder.
        embedding_dim = 100

        embedding_matrix = np.zeros((max_words, embedding_dim))
        for word, i in word_index.items():
            if i < max_words:
                embedding_vector = embeddings_idx.get(word)
                if embedding_vector is not None:
                    # Words not encountered in the embedding index will be given all zeros
                    embedding_matrix[i] = embedding_vector

        # Now let's build a simple model and set the weights of the embedding layer using the matrix we just generated
        model = models.Sequential()
        model.add(layers.Embedding(max_words, embedding_dim, input_length=max_len))
        model.add(layers.Flatten())
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))

        # Freeze the embedding layer to prevent forgetting the pre-learned embedding space
        model.layers[0].set_weights([embedding_matrix])
        model.layers[0].trainable = False

        # Compile and train the model
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
        model.summary()

        history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))

        # Save the model
        model.save('C:\\Datasets\\IMDB\\pretrained_glove_model.h5')

        # Plot the training and validation accuracy and loss
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(1, len(acc) + 1)

        plt.plot(epochs, acc, 'bo', label='Training Acc')
        plt.plot(epochs, val_acc, 'b', label='Validation Acc')
        plt.title('Training and Validation Accuracy vs. Epochs')
        plt.legend()

        plt.figure()

        plt.plot(epochs, loss, 'bo', label='Training Loss')
        plt.plot(epochs, val_loss, 'b', label='Validation Loss')
        plt.title('Training and Validation Loss vs. Epochs')
        plt.legend()

        plt.show()

        # Now for testing purposes, lets compile the test data from the dataset and evaluate the model
        test_dir = os.path.join(imdb_dir, 'test')

        labels = []
        texts = []

        for label_type in ['neg', 'pos']:
            dir_name = os.path.join(test_dir, label_type)
            for fname in sorted(os.listdir(dir_name)):
                if fname[-4:] == '.txt':
                    f = open(os.path.join(dir_name, fname), encoding="utf8")
                    texts.append(f.read())
                    f.close()
                    if label_type == 'neg':
                        labels.append(0)
                    else:
                        labels.append(1)

        sequences = tokeniser.texts_to_sequences(texts)
        x_test = preprocessing.sequence.pad_sequences(sequences, max_len)
        y_test = np.asarray(labels)

        # Evaluate the model
        model.evaluate(x_test, y_test)

    def rnn_example():
        """
        A function which briefly goes through how to implement simple Recurrent Neural Networks using the IMDB dataset
        as an example

        :return: None
        """
        # We'll build a very simple model first using an embedding layer and an RNN layer, just to show how to import
        # these layers from Keras.layers. Pay close attention to the output shape of the RNN layer.
        model = models.Sequential()
        model.add(layers.Embedding(10000, 32))
        model.add(layers.SimpleRNN(32))
        model.summary()
        model = None

        # Now we'll show an example where there are multiple RNN layers in sequence. Note first that it is now required
        # to return the full output sequence for each hidden RNN layer, however the final output layer must only return
        # it's last output value.
        model = models.Sequential()
        model.add(layers.Embedding(10000, 32))
        model.add(layers.SimpleRNN(32, return_sequences=True))
        model.add(layers.SimpleRNN(32, return_sequences=True))
        model.add(layers.SimpleRNN(32, return_sequences=True))
        model.add(layers.SimpleRNN(32, return_sequences=True))
        model.add(layers.SimpleRNN(32))
        model.summary()
        model = None

        # Now let's try processing the IMDB dataset using this simple network. Let's load the data and pad the input
        # sequences to the size required
        max_features = 10000
        max_len = 500
        batch_size = 128

        print("\nLoading data...")
        (input_train, output_train), (input_test, output_test) = imdb.load_data(num_words=max_features)
        print(f"{len(input_train)} training sequences")
        print(f"{len(input_test)} testing sequences")

        print("Pad sequences (samples x time)")
        input_train = preprocessing.sequence.pad_sequences(input_train, max_len)
        input_test = preprocessing.sequence.pad_sequences(input_test, max_len)
        print(f"input_train shape: {input_train.shape}")
        print(f"input_test shape: {input_test.shape}")

        # Now we'll build a small and simple model using the SimpleRNN layer to classify the IMDB reviews
        model = models.Sequential()
        model.add(layers.Embedding(max_features, 32))
        model.add(layers.SimpleRNN(32))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
        model.summary()

        # Now train the model and plot its performance
        history = model.fit(input_train, output_train, epochs=10, batch_size=batch_size, validation_split=0.2)

        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(1, len(acc) + 1)

        plt.plot(epochs, acc, 'bo', label='Training Acc')
        plt.plot(epochs, val_acc, 'b', label='Validation Acc')
        plt.title('Training and Validation Accuracy vs. Epochs')
        plt.legend()

        plt.figure()

        plt.plot(epochs, loss, 'bo', label='Training Loss')
        plt.plot(epochs, val_loss, 'b', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        plt.show()

    rnn_example()
