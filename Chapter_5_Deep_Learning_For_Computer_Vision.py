"""
Chapter 5: Deep Learning for Computer Vision

This chapter covers:
    -> Understanding Convolutional Neural Networks
    -> Using Data Augmentation to mitigate overfitting
    -> Using a pre-trained convnet to do feature extraction
    -> visualising what convnets learn and how they make classification decisions
"""

import os
import shutil
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models, layers, optimizers

if __name__ == '__main__':
    """
    Run using `python -m Chapter_5_Deep_Learning_For_Computer_Vision`
    """

    def mnist_conv_net():
        print("Let's first take a look at what a typical convnet looks like:")

        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))

        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        model.summary()

        print(
            "This particular model is actually quite useful for the mnist dataset we encountered in Chapter 2. As before,"
            "we'll normalise the inputs and categorise the outputs.")

        # Load data and print useful information
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        print(f"There are {len(train_images)} training images, each of shape {train_images[0].shape}: dtype = "
              f"{train_images[0].dtype},")
        print(f"and {len(test_images)} training images, each of shape {test_images[0].shape}: dtype = "
              f"{test_images[0].dtype}\n")

        # Reshape and normalise the inputs for use in the neural network
        print('Reshaping and normalising inputs:')
        train_images = train_images.reshape((len(train_images), 28, 28, 1))
        train_images = train_images.astype('float32') / 255

        test_images = test_images.reshape((len(test_images), 28, 28, 1))
        test_images = test_images.astype('float32') / 255

        print(f"Now there are {len(train_images)} training images, each of shape {train_images[0].shape}: dtype = "
              f"{train_images[0].dtype},")
        print(f"and {len(test_images)} training images, each of shape {test_images[0].shape}: dtype = "
              f"{test_images[0].dtype}\n")

        # Convert the outputs to categorical labels which are more useful for training
        print('Converting the outputs to categorical labels:')
        train_labels = to_categorical(train_labels)
        test_labels = to_categorical(test_labels)

        # Train the model and evaluate the performance
        print("\nNow let's train and evaluate the model to see how it performs on the 1st try:")
        model.fit(train_images, train_labels, epochs=5, batch_size=64)

        test_loss, test_acc = model.evaluate(test_images, test_labels)
        print(f"Test Accuracy = {test_acc * 100}%\nNote the immediate rise in test accuracy. This fairly naive network "
              f"can easily outperform simple Dense networks due to convnets' inherent resistance to translational and spatial "
              f"variations. But what more can convnets do?")

    def cats_vs_dogs():
        original_dataset_dir = 'C:\\Datasets\\dogs-vs-cats\\train'

        base_dir = 'C:\\Datasets\\dogs-vs-cats\\cats_and_dogs_small'

        train_dir = os.path.join(base_dir, 'train')
        validation_dir = os.path.join(base_dir, 'validation')
        test_dir = os.path.join(base_dir, 'test')

        def mk_smaller_dataset():
            """
            Use this function to take a subset of the dataset for the dogs vs cats competition and save it elsewhere
            under labelled folders

            :return: None
            """
            os.mkdir(base_dir)
            os.mkdir(train_dir)
            os.mkdir(validation_dir)
            os.mkdir(test_dir)

            train_cats_dir = os.path.join(train_dir, 'cats')
            os.mkdir(train_cats_dir)
            train_dogs_dir = os.path.join(train_dir, 'dogs')
            os.mkdir(train_dogs_dir)

            validation_cats_dir = os.path.join(validation_dir, 'cats')
            os.mkdir(validation_cats_dir)
            validation_dogs_dir = os.path.join(validation_dir, 'dogs')
            os.mkdir(validation_dogs_dir)

            test_cats_dir = os.path.join(test_dir, 'cats')
            os.mkdir(test_cats_dir)
            test_dogs_dir = os.path.join(test_dir, 'dogs')
            os.mkdir(test_dogs_dir)

            fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
            for fname in fnames:
                src = os.path.join(original_dataset_dir, fname)
                dst = os.path.join(train_cats_dir, fname)
                shutil.copyfile(src, dst)

            fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
            for fname in fnames:
                src = os.path.join(original_dataset_dir, fname)
                dst = os.path.join(validation_cats_dir, fname)
                shutil.copyfile(src, dst)

            fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
            for fname in fnames:
                src = os.path.join(original_dataset_dir, fname)
                dst = os.path.join(test_cats_dir, fname)
                shutil.copyfile(src, dst)

            fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
            for fname in fnames:
                src = os.path.join(original_dataset_dir, fname)
                dst = os.path.join(train_dogs_dir, fname)
                shutil.copyfile(src, dst)

            fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
            for fname in fnames:
                src = os.path.join(original_dataset_dir, fname)
                dst = os.path.join(validation_dogs_dir, fname)
                shutil.copyfile(src, dst)

            fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
            for fname in fnames:
                src = os.path.join(original_dataset_dir, fname)
                dst = os.path.join(test_dogs_dir, fname)
                shutil.copyfile(src, dst)

        def train_model_without_augmentation():
            model = models.Sequential()
            model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Conv2D(64, (3, 3), activation='relu'))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Conv2D(128, (3, 3), activation='relu'))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Conv2D(128, (3, 3), activation='relu'))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Flatten())
            model.add(layers.Dense(512, activation='relu'))
            model.add(layers.Dense(1, activation='sigmoid'))

            model.summary()
            model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

            train_datagen = ImageDataGenerator(rescale=1. / 255)
            test_datagen = ImageDataGenerator(rescale=1. / 255)

            train_generator = train_datagen.flow_from_directory(train_dir,
                                                                target_size=(150, 150),
                                                                batch_size=20,
                                                                class_mode='binary')
            validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                                    target_size=(150, 150),
                                                                    batch_size=20,
                                                                    class_mode='binary')
            history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=30,
                                          validation_data=validation_generator, validation_steps=50)

            model.save(filepath='C:\\Datasets\\dogs-vs-cats\\cats_and_dogs_small\\cats_and_dogs_small_1.h5')

            acc = history.history['acc']
            val_acc = history.history['val_acc']
            loss = history.history['loss']
            val_loss = history.history['val_loss']

            epochs = range(1, len(acc) + 1)

            plt.plot(epochs, acc, 'bo', label='Training acc')
            plt.plot(epochs, val_acc, 'b', label='Validation acc')
            plt.title('Training and Validation Accuracy vs. Epoch #')
            plt.legend()

            plt.figure()

            plt.plot(epochs, loss, 'bo', label='Training loss')
            plt.plot(epochs, val_loss, 'b', label='Validation loss')
            plt.title('Training and Validation Loss vs. Epoch #')
            plt.legend()

            plt.show()

    cats_vs_dogs()
