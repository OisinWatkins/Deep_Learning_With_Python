"""
Chapter 2: The mathematical building blocks of neural networks.

This chapter covers:
    -> A first example of a Neural Network
    -> Tensors and Tensor Operations
    -> How Neural Networks learn via backpropagation and gradient descent.
"""

import matplotlib.pyplot as plt
from tensorflow.keras import models, layers
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

if __name__ == '__main__':
    """
    Run using `python -m Chapter_2_NN_fundamentals`
    """

    # Load data and print useful information:
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    print(f"There are {len(train_images)} training images, each of shape {train_images[0].shape}: dtype = "
          f"{train_images[0].dtype},")
    print(f"and {len(test_images)} training images, each of shape {test_images[0].shape}: dtype = "
          f"{test_images[0].dtype}\n")

    # Display an example image
    digit = train_images[4]
    plt.imshow(digit)
    plt.show()

    # Reshape and normalise the inputs for use in the neural network
    print('Reshaping and normalising inputs:')
    train_images = train_images.reshape((len(train_images), train_images[0].shape[0] * train_images[0].shape[1]))
    train_images = train_images.astype('float32') / 255

    test_images = test_images.reshape((len(test_images), test_images[0].shape[0] * test_images[0].shape[1]))
    test_images = test_images.astype('float32') / 255

    print(f"Now there are {len(train_images)} training images, each of shape {train_images[0].shape}: dtype = "
          f"{train_images[0].dtype},")
    print(f"and {len(test_images)} training images, each of shape {test_images[0].shape}: dtype = "
          f"{test_images[0].dtype}\n")

    # Convert the outputs to categorical labels which are more useful for training
    print('Converting the outputs to categorical labels:')
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    # Build the network
    print('Building network outline')
    network = models.Sequential()
    network.add(layers.Dense(units=512, activation='relu', input_shape=(28*28,), kernel_constraint=max_norm()))
    network.add(layers.Dense(10, activation='softmax', kernel_constraint=max_norm()))

    network.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    # Train and test the network, printing the results for the user
    network.fit(train_images, train_labels, epochs=5, batch_size=128)
    test_loss, test_acc = network.evaluate(test_images, test_labels)
    print(f"Test Accuracy: {test_acc * 100}%")
