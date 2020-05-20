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
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

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

            # Extract the names of each cat jpg at index 0 -> 1000 and move to a dedicated folder
            fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
            for fname in fnames:
                src = os.path.join(original_dataset_dir, fname)
                dst = os.path.join(train_cats_dir, fname)
                shutil.copyfile(src, dst)

            # Extract the names of each cat jpg at index 1000 -> 1500 and move to a dedicated folder
            fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
            for fname in fnames:
                src = os.path.join(original_dataset_dir, fname)
                dst = os.path.join(validation_cats_dir, fname)
                shutil.copyfile(src, dst)

            # Extract the names of each cat jpg at index 1500 -> 2000 and move to a dedicated folder
            fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
            for fname in fnames:
                src = os.path.join(original_dataset_dir, fname)
                dst = os.path.join(test_cats_dir, fname)
                shutil.copyfile(src, dst)

            # Extract the names of each dog jpg at index 0 -> 1000 and move to a dedicated folder
            fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
            for fname in fnames:
                src = os.path.join(original_dataset_dir, fname)
                dst = os.path.join(train_dogs_dir, fname)
                shutil.copyfile(src, dst)

            # Extract the names of each dog jpg at index 1000 -> 1500 and move to a dedicated folder
            fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
            for fname in fnames:
                src = os.path.join(original_dataset_dir, fname)
                dst = os.path.join(validation_dogs_dir, fname)
                shutil.copyfile(src, dst)

            # Extract the names of each dog jpg at index 1500 -> 2000 and move to a dedicated folder
            fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
            for fname in fnames:
                src = os.path.join(original_dataset_dir, fname)
                dst = os.path.join(test_dogs_dir, fname)
                shutil.copyfile(src, dst)

        def train_model_without_augmentation():
            """
            This function will train a fairly naive network using the segmented datasets we generated using
            mk_smaller_dataset() and plot the training and validation performance after training is complete.

            :return: None
            """
            # Layout a pretty simple network and compile with binary crossentropy.
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

            # Create generates dedicate to provide scaled image inputs to the network.
            train_datagen = ImageDataGenerator(rescale=1. / 255)
            test_datagen = ImageDataGenerator(rescale=1. / 255)

            # One generator for training and one for validation
            train_generator = train_datagen.flow_from_directory(train_dir,
                                                                target_size=(150, 150),
                                                                batch_size=20,
                                                                class_mode='binary')
            validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                                    target_size=(150, 150),
                                                                    batch_size=20,
                                                                    class_mode='binary')

            # Train the model using the dedicated fit_generator() function [I think this is outdated now, but this is
            # how the book does it]
            history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=30,
                                          validation_data=validation_generator, validation_steps=50)

            # Save the trained model and plot the training and validation performance
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

        def train_model_with_augmentation():
            """
            This function will train a more sophisticated network using the segmented datasets we generated using
            mk_smaller_dataset() and augmentation and plot the training and validation performance after training is
            complete.

            :return: None
            """
            # Layout a network with Dropout this time and compile with binary crossentropy.
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
            model.add(layers.Dropout(0.5))
            model.add(layers.Dense(512, activation='relu'))
            model.add(layers.Dense(1, activation='sigmoid'))

            model.summary()
            model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

            # Create generators for both training and validation. Give the training generator augmentation functionality
            train_datagen = ImageDataGenerator(rescale=1. / 255,
                                               rotation_range=40,
                                               width_shift_range=0.2,
                                               height_shift_range=0.2,
                                               shear_range=0.2,
                                               zoom_range=0.2,
                                               horizontal_flip=True)

            test_datagen = ImageDataGenerator(rescale=1. / 255)

            train_generator = train_datagen.flow_from_directory(train_dir,
                                                                target_size=(150, 150),
                                                                batch_size=20,
                                                                class_mode='binary')
            validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                                    target_size=(150, 150),
                                                                    batch_size=20,
                                                                    class_mode='binary')

            # Train the model. More epochs will be needed this time for 2 reasons:
            #   -> Dropout will greatly increase time to convergence
            #   -> Augmentation results in effectively infinite input variation.
            history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=100,
                                          validation_data=validation_generator, validation_steps=50)

            # Save the model and plot the training and validation performance
            model.save(filepath='C:\\Datasets\\dogs-vs-cats\\cats_and_dogs_small\\cats_and_dogs_small_2.h5')

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

        def reuse_conv_base():
            """
            This function will reuse the bottom of the VGG16 network to extract the features of the images and then use
            a new classifier to perform the classification. The issue with this approach is that the classifier overfits
            almost immediately. The chapter in the book goes into detail into how this can be avoided by incorporating
            the base into the model and training end - to - end with augmentation, however this is too expensive to run
            on CPU and is essentially a mix of train_model_with_augmentation() and reuse_conv_base()

            :return: None
            """
            # Extract the convolutional base of VGG16 and print the summary
            conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
            conv_base.summary()

            # Prepare the image generator
            datagen = ImageDataGenerator(rescale=1./255)
            batch_size = 20

            def extract_features(directory, sample_count):
                """
                Simple function to pass the inputs from the dataset through the convolutional base to extract their
                features

                :param directory: Directory to the dataset of interest
                :param sample_count: Number of samples to preprocess
                :return: The processed features and their labels
                """
                features = np.zeros(shape=(sample_count, 4, 4, 512))
                labels = np.zeros(shape=(sample_count))

                generator = datagen.flow_from_directory(directory,
                                                        target_size=(150, 150),
                                                        batch_size=batch_size,
                                                        class_mode='binary')

                i = 0
                for inputs_batch, labels_batch in generator:
                    features_batch = conv_base.predict(inputs_batch)
                    features[i * batch_size: (i + 1) * batch_size] = features_batch
                    labels[i * batch_size: (i + 1) * batch_size] = labels_batch

                    i += 1
                    if i * batch_size >= sample_count:
                        break
                return features, labels

            # Pre-process data for training, validation and testing
            train_features, train_labels = extract_features(train_dir, 2000)
            validation_features, validation_labels = extract_features(validation_dir, 1000)
            test_features, test_labels = extract_features(test_dir, 1000)

            train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
            validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
            test_features = np.reshape(test_features, (1000, 4 * 4 * 512))

            # Layout the classifier (very simple) and train. Record the training and validation performance
            model = models.Sequential()
            model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
            model.add(layers.Dropout(0.5))
            model.add(layers.Dense(1, activation='sigmoid'))

            model.compile(optimizer=optimizers.RMSprop(lr=2e-5), loss='binary_crossentropy', metrics=['acc'])

            history = model.fit(train_features, train_labels, epochs=30, batch_size=20,
                                validation_data=(validation_features, validation_labels))

            # Save the model and plot the performance
            model.save(filepath='C:\\Datasets\\dogs-vs-cats\\cats_and_dogs_small\\cats_and_dogs_small_ReuseBottom.h5')

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

        def visualise_convnet():
            """
            This function details how to visualise what exactly a convnet learns. First we'll load the model trained
            with augmentation and plot the outputs from individual layers as images. Second we'll use the VGG16 network
            and use gradient ascent in the input space to show what each filter maximally responds to.

            -> The filter patterns section does not seem to work, I believe the patterns are actually overlapping,
               causing everything to white out

            :return: None
            """
            # Load the model and print a brief summary
            model = load_model('C:\\Datasets\\dogs-vs-cats\\cats_and_dogs_small\\cats_and_dogs_small_Aug.h5')
            model.summary()

            # Grab a picture and convert it to a scaled tensor.
            img_path = 'C:\\Datasets\\dogs-vs-cats\\cats_and_dogs_small\\test\\cats\\cat.1700.jpg'
            img = image.load_img(img_path, target_size=(150, 150))
            img_tensor = image.img_to_array(img)
            img_tensor = np.expand_dims(img_tensor, axis=0)
            img_tensor /= 255

            # print(img_tensor.shape)
            # plt.imshow(img_tensor[0])
            #
            # plt.figure()

            # Create a new model who's outputs are the layer activations of the original model. Then compute the
            # activations using the input image tensor.
            layer_outputs = [layer.output for layer in model.layers[:8]]
            activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

            activations = activation_model.predict(img_tensor)
            # first_layer_activation = activations[0]
            # print(first_layer_activation.shape)
            #
            # plt.matshow(first_layer_activation[0, :, :, 7], cmap='viridis')
            # plt.show()

            # Extract the names of each layer
            layer_names = []
            for layer in model.layers[:8]:
                layer_names.append(layer.name)

            print(layer_names)

            # Before performing the plotting, inform numpy that divide by invalid values is to be ignored.
            np.seterr(divide='ignore', invalid='ignore')

            # Iteratively plot each layer's activations, one channel at a time.
            images_per_row = 16
            for layer_name, layer_activation in zip(layer_names, activations):
                n_features = layer_activation.shape[-1]
                size = layer_activation.shape[1]

                n_cols = n_features // images_per_row
                display_grid = np.zeros((size * n_cols, images_per_row * size))

                for col in range(n_cols):
                    for row in range(images_per_row):
                        channel_image = layer_activation[0, :, :, col * images_per_row + row]
                        channel_image -= channel_image.mean()
                        channel_image /= channel_image.std()
                        channel_image *= 64
                        channel_image += 128
                        channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                        display_grid[col * size: (col + 1) * size, row * size: (row + 1) * size] = channel_image

                scale = 1. / size
                plt.figure(figsize=(scale * display_grid.shape[1],
                                    scale * display_grid.shape[0]))
                plt.title(layer_name)
                plt.grid(False)
                plt.imshow(display_grid, aspect='auto')

            plt.show()

            # ----------------------------------------------------------------------------------------------------------
            # ----------------------------------------------------------------------------------------------------------

            # Now we'll look at filter activations another way. Oreviously we viewed how a network's layers responded to
            # a sample image. Now we'll use `gradient ascent in the input space` to examine what each filter in the
            # VGG16 network maximally responds to.
            new_model = VGG16(weights='imagenet', include_top=False)

            # The resulting image may have illegal values, hence we need to post-process the result
            def deprocess_image(x):
                # Make x zero-centered with std = 0.1
                x -= x.mean()
                x /= x.std()
                x *= 0.1

                # Clip values to [0, 1]
                x += 0.5
                x = np.clip(x, 0, 1)

                # Convert to RGB array
                x *= 255
                x = np.clip(x, 0, 255).astype('uint8')
                return x

            # Create a function that will maximise the activation of the nth filter in a given network layer
            def generate_pattern(new_layer_name, filter_index, Size=150):
                new_layer_output = new_model.get_layer(new_layer_name).output
                loss = K.mean(new_layer_output[:, :, :, filter_index])

                # Compute the gradients of the loss function w.r.t. the network inputs. Scale using the L2 norm
                grads = K.gradients(loss, new_model.input)[0]
                grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

                # Define a Keras backend function to compute the loss tensor and the gradient tensor given an input image
                iterate = K.function([new_model.input], [loss, grads])
                loss_value, grads_value = iterate([np.zeros((1, Size, Size, 3))])

                # At this point we can define a Python loop to perform Stochastic Gradient Descent
                # Start with a grey image with some noise
                input_img_data = np.random.random((1, Size, Size, 3)) * 20 + 128

                # define the magnitude of each gradient step
                step = 1
                for a in range(40):
                    # Compute the loss and gradient for each input
                    loss_value, grads_value = iterate([input_img_data])

                    # Update the input image along the positive gradient to maximise the loss
                    input_img_data += grads_value * step

                Image = input_img_data[0]
                return deprocess_image(Image)

            plt.imshow(generate_pattern('block3_conv1', 0))
            plt.show()

            # Now we'll use this function to plot the maximal response of each filter an a layer
            layer_name = 'block3_conv1'
            size = 64
            margin = 5

            # Create a black array to store the image results
            results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * size, 3))

            for i in range(8):
                for j in range(8):
                    filter_img = generate_pattern(layer_name, i + (j * 8), Size=size)

                    horizontal_start = i * size + i * margin
                    horizontal_end = horizontal_start + size
                    vertical_start = j * size + j * margin
                    vertical_end = vertical_start + size
                    results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img

            # Plot the resulting image
            plt.figure(figsize=(20, 20))
            plt.imshow(results)
            plt.show()

        train_model_without_augmentation()
        train_model_with_augmentation()
        reuse_conv_base()
        visualise_convnet()

    def visualise_heatmaps():
        """
        This function uses the VGG16 network and processes an images stored in the project folder of 2 African Elephants.
        The intent of the code is to process the image and then determine the gradient of the African Elephant class
        w.r.t the last convolutional layer of the network. I couldn't get the overlay section working as many of the
        libraries used in the book have since been outdated.

        :return: None
        """
        model = VGG16(weights='imagenet')
        img_path = 'C:\\Users\\owatkins\\OneDrive - Analog Devices, Inc\\Documents\\Project Folder\\Tutorials and ' \
                   'Courses\\Deep Learning with Python\\African-Elephant-With-Baby.jpg'
        # Load the image and plot it
        img = image.load_img(img_path, target_size=(224, 224))
        plt.imshow(img)
        plt.show()

        # Convert the image to a numpy array and reshape
        x_arr = image.img_to_array(img)
        x = np.expand_dims(x_arr, axis=0)
        x = preprocess_input(x)

        # Perform the forward pass using the VGG16 network. Print the top 3 predictions and their probabilities, as well
        # as the index of the most prominent output.
        preds = model.predict(x)
        print('Predictions: ', decode_predictions(preds, top=3)[0])
        idx_of_pred = np.argmax(preds[0])
        print('Index of the max prediction: ', idx_of_pred)

        # African Elephant Entry in the prediction vector, and the output feature map of the last convolutional layer
        african_elephant_output = model.output[:, 386]
        last_conv_layer = model.get_layer('block5_conv3')

        # Gradient of the African Elephant class w.r.t. the output feature map. Each of the (512, ) entries from
        # pooled_grads is the mean intensity of the gradient over a specific feature map-channel
        grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]
        pooled_grads = K.mean(grads, axis=(0, 1, 2))

        # Function to let you access the values of the quantities just defined: pooled_grads and the output feature map
        # of block5_conv3 given the input image
        iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
        pooled_grads_value, conv_layer_output_value = iterate([x])

        # Multiply each channel in the feature map byt "how important" that channel is w.r.t. the elephant class
        for i in range(512):
            conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

        # Compute the channel-wise mean of the resulting heatmap of class activation and plot.
        heatmap = np.mean(conv_layer_output_value, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        plt.matshow(heatmap)
        plt.show()

    mnist_conv_net()
    cats_vs_dogs()
    visualise_heatmaps()
