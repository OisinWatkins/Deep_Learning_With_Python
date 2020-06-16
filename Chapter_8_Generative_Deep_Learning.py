"""
Chapter 8: Generative Deep Learning

This chapter covers:
    -> Text Generation with LSTM
    -> Implementing DeepDream
    -> Performing Natural Style Transfer
    -> Variational Autoencoders
    -> Understanding Generative Adversarial Networks
"""
import os
import sys
import time
from scipy.optimize import fmin_l_bfgs_b
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.stats import norm
from tensorflow import keras
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras import models, layers
from tensorflow.keras.applications import inception_v3, vgg19
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image

if __name__ == '__main__':
    """
    run using `python -m Chapter_8_Generative_Deep_Learning`
    """

    def text_generation():
        """
        This function will explore the use of LSTM's for text generation.

        :return: None
        """
        # It is important to read "The importance of Sampling Strategy" before continuing with this code. We'll use an
        # LSTM with a softmax output layer to predict the next token (in some case a character, in others a word) in a
        # sequence. Using 'greedy sampling' (where you simply take the most likely next token) will usually result in
        # less interesting and repetitive sequences being generated, so we'll re-weight the softmax output using a
        # temperature control.

        def rewrite_distribution(original_distribution, temperature=0.5):
            """

            :param original_distribution: 1D numpy array of probability values that must sum to 1.
            :param temperature: A factor value controlling the entropy of the output distribution
            :return: A re-weighted version of the input distribution.
            """

            # Divide the original distribution by the temperature factor and raise e to every value in the distribution,
            # thereby creating a new weighted distribution.
            distribution = np.log(original_distribution) / temperature
            distribution = np.exp(distribution)

            # Divide the distribution by its sum to ensure the distribution sums to 1.
            return distribution / np.sum(distribution)

        # Now let's implement a character-level text generation model. The first thing we'll need in a large body of
        # text data. A corpus of Nietzsche's works from the late 19th century should be enough. The result model will
        # learn to generate text in the style of Nietzsche.
        path = keras.utils.get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
        text = open(path).read().lower()
        print(f"\nCorpus Length: {len(text)}")

        # Now we need to encode the inputs and training outputs into Numpy arrays. We'll use one-hot encoding on a
        # character level and generate sequences which are maxlen long. We'll sample a new sequence every step
        # characters.
        maxlen = 60
        step = 3

        sentences = []
        next_chars = []
        for i in range(0, len(text) - maxlen, step):
            sentences.append(text[i: i + maxlen])
            next_chars.append(text[i + maxlen])

        print(f"Number of Sequences: {len(sentences)}")

        # Now extract the set of unique characters in the corpus. Store the resulting list as a dictionary where each
        # character is stored alongside its index.
        chars = sorted(list(set(text)))
        print(f"Number of Unique Characters: {len(chars)}")
        chars_indices = dict((char, chars.index(char)) for char in chars)

        # Now we one-hot encode the characters into binary arrays.
        print("Vectorizing...")
        x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
        y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                x[i, t, chars_indices[char]] = 1
            y[i, chars_indices[next_chars[i]]] = 1

        # Now to build the network. In this example we'll use an LSTM, however 1D convnets have proven to be
        # exceptionally good at this task while being a lightweight alternative.
        model = models.Sequential()
        model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))
        model.add(layers.Dense(len(chars), activation='softmax'))

        optimizer = keras.optimizers.RMSprop(lr=0.01)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        model.summary()

        # Given the specificities of this particular task we have to define our own training loop to do the following:
        #   1) Draw from the model a probability distribution for the next character, given the generated text so far.
        #   2) Rewrite the distribution to a certain temperature.
        #   3) Sample the next character at random according to the re-weighted distribution.
        #   4) Add the new character at the end of the available text.
        def sample(preds, temperature=1.0):
            """
            This function will take the softmax output from the model and use them to randomly sample our character
            space according to a re-weighted distribution.

            :param preds: Predictions provided by the softmax output
            :param temperature: Factor controlling the entropy of our re-weighted predictions
            :return: The max of a random array created according to a re-weighted version of the softmax outputs
            """
            preds = np.asarray(preds).astype('float64')
            preds = rewrite_distribution(preds, temperature)
            probas = np.random.multinomial(1, preds, 1)
            return np.argmax(probas)

        # Now we have to manually train the network. Interestingly doing so manually allows us to see how the model
        # learns over time, as well as monitor the impact of the temperature factor.
        for epoch in range(1, 60):
            print(f"\n{epoch} Epoch")
            # Train for 1 epoch
            model.fit(x, y, batch_size=128, epochs=1)

            # provide the seed information for the model's prediction
            start_index = np.random.randint(0, len(text) - maxlen - 1)
            generated_text = text[start_index: start_index + maxlen]
            print('\n--- Generating with seed: "' + generated_text + '"')

            # Now examine the effect of the temperature factor
            for temperature in [0.2, 0.5, 1.0, 1.2]:
                # Print the current temperature value and the seed text.
                print('\n------ temperature: ', temperature)
                sys.stdout.write(generated_text)

                for i in range(400):
                    # Generate the next 400 characters worth of text using the model's predictions.
                    sampled = np.zeros((1, maxlen, len(chars)))
                    for t, char in enumerate(generated_text):
                        sampled[0, t, chars_indices[char]] = 1

                    preds = model.predict(sampled, verbose=0)[0]
                    next_index = sample(preds, temperature)
                    next_char = chars[next_index]

                    # Always be certain to both update the generated text and keep the generated text value's length
                    # constant.
                    generated_text += next_char
                    generated_text = generated_text[1:]

                    # Append the next character to the generated text.
                    sys.stdout.write(next_char)

        # Save the model.
        model.save(filepath='C:\\Datasets\\Nietzsche\\text_generation_model.h5')

    """
    We're going to define a bunch of auxiliary functions before using either deep_dream() or neural_style_transfer
    """
    def resize_img(img, size):
        """
        Handles resizing the image using Pillow, deprocess_image and numpy.

        :param img: Image
        :param size: Desired Size
        :return: The input image resized to the desired size as np array.
        """
        pil_img = deprocess_image_inception(img)
        new_size = (1, int(size[0]), int(size[1]), 3)
        return np.array(img).resize(new_size, refcheck=False)

    def save_img(img, fname, inception=True):
        """
        Saves the image using the Pillos.Image library and the deprocess_image function.

        :param img: Image as numpy array
        :param fname: Filename for the new image
        :return: None
        """
        if inception:
            pil_image = Image.fromarray(deprocess_image_inception(np.copy(img)))
        else:
            pil_image = Image.fromarray(deprocess_image_vgg19(np.copy(img)))
        pil_image.save(fname)

    def preprocess_image_inception(image_path):
        """
        Preprocess the image using the inception_v3 preprocess function.

        :param image_path: Path to the image file.
        :return: The preprocessed image using inception_v3 packaged function.
        """
        img = image.load_img(image_path)
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = inception_v3.preprocess_input(img)
        return img

    def preprocess_image_vgg19(image_path, img_h, img_w):
        """
        Preprocess the image using the VGG19 preprocess function.

        :param image_path: Path to the image
        :param img_h: The height of the desired processed image
        :param img_w: The width of the desired processed image
        :return: The preprocessed img using the VGG19 network
        """
        img = image.load_img(image_path, target_size=(img_h, img_w))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = vgg19.preprocess_input(img)
        return img

    def deprocess_image_inception(x):
        """
        This function takes a normalised image input and deprocesses it to image compatible values.

        :param x: A normalised input tensor representing an image.
        :return: An image-viewer compatible version of the input
        """
        if K.image_data_format() == 'channels_first':
            x = x.reshape((3, x.shape[2], x.shape[3]))
            x = x.transpose((1, 2, 0))
        else:
            x = x.reshape((x.shape[1], x.shape[2], 3))

        x /= 2.0
        x += 0.5
        x *= 255.0
        x = np.clip(x, 0, 255).astype('uint8')
        return Image.fromarray(x)

    def deprocess_image_vgg19(x):
        """
        This function takes a normalised image input and deprocesses it to image compatible values.

        :param x: A normalised input tensor representing an image.
        :return: An image-viewer compatible version of the input

        :param x:
        :return:
        """
        x = x.astype('float64')
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68
        x = x[:, :, ::-1]
        x = np.clip(x, 0, 255).astype('uint8')
        return x

    """
    End of auxiliary functions
    """

    def deep_dream():
        """
        DeepDream is an artistic image-modification technique that uses the representations learned by convnets. First
        released by Google in the summer of 2015, this algorithm is very similar to the gradient ascent technique we
        viewed earlier to represent the patterns learned by individual filters during training (Chapter 5). There are a
        few differences to the algorithm:
            -> With DeepDream you try to maximise the activation of the entire layer rather than one specific filter,
               thus mixing together visualisations of a larger number of filters.
            -> You start not from a blank, slightly noisy input, but rather from an existing image - thus the resulting
               effects latch on to preexisting visual patterns, distorting elements of the image in a somewhat artistic
               fashion.
            -> The input images are processed at different scales (called octaves), which improves the quality of the
               visualisations.
               
               
        This function does not work due to version issues.

        :return: None
        """

        # You won't be training a model for this application, so let's disable all training functionality before
        # starting
        K.set_learning_phase(0)

        model = inception_v3.InceptionV3(weights='imagenet', include_top=False)

        # In Chapter 5 we use the loss value to maximise the output of a specific filter. This time we'll attempt to
        # maximise the weighted sum of the L2 norm of the activations of a set of high-level layers. The set of layers
        # chosen will have a massive impact on the resulting modifications to the image, so make these params very
        # easily configurable.
        layers_contributions = {'mixed2': 0.2,
                                'mixed3': 3.0,
                                'mixed4': 2.0,
                                'mixed5': 1.5}
        layer_dict = dict([(layer.name, layer) for layer in model.layers])

        # You'll define the loss by adding layer contributions to this scalar value.
        loss = K.variable(0.0)
        for layer_name in layers_contributions:
            coeff = layers_contributions[layer_name]
            # Retrieve the layer's output.
            activation = layer_dict[layer_name].output

            # Define the scaling factor and add the L2 norm of the features of a layer to the loss. You avoid boarder
            # artifacts by involving non-boarder pixels in the loss.
            scaling = K.prod(K.cast(K.shape(activation), 'float32'))
            loss = loss + coeff * K.sum(K.square(activation[:, 2: -2, 2: -2, :])) / scaling

        # Now we can set up the gradient ascent process.
        dream = model.input

        # Compute gradient of the dream w.r.t to the loss, then NORMALISE!!!
        grads = K.gradients(loss, dream)[0]
        grads /= K.minimum(K.mean(K.abs(grads)), 1e-7)

        # Now set up a Keras function to retrieve the value of the loss and gradients given an input image.
        outputs = [loss, grads]
        fetch_loss_and_grads = K.function([dream], outputs)

        def eval_loss_and_grads(x):
            """
            This function is used to call the fetch_loss_and_grads function and package the outputs in an easy to use
            fashion.

            :param x: Input dream
            :return: The loss and the gradient of the layer w.r.t. the dream.
            """
            outs = fetch_loss_and_grads([x])
            loss_value = outs[0]
            grads_value = outs[1]
            return loss_value, grads_value

        def gradient_ascent(x, iterations, step, max_loss=None):
            """
            This function runs gradient ascent for a number of iterations.

            :param x: Input dream
            :param iterations: Number of iterations to run gradient ascent for
            :param step: Step-size of the gradient ascent
            :param max_loss: Maximum loss we'll accept during the gradient ascent before stopping.
            :return: A modified version of the input dream
            """
            for i in range(iterations):
                loss_value, grads_value = eval_loss_and_grads(x)
                if max_loss is not None and loss_value > max_loss:
                    break
                print(f"...Loss value at {i}: {loss_value}")
                x += step * grads_value
            return x

        # Now we can begin programming the DeepDream algorithm itself. First we need to define a set of scales
        # (called octaves) at which to process the image. Each octave is 40% larger than the last. At each scale (from
        # smallest to largest) you run gradient ascent to maximise the loss you previously defined. To prevent artifacts
        # of up-scaling (blurriness and stretching) we'll re-inject the lost back into the image, which is possible
        # because you know what the original image should look like at a larger scale.
        step = 0.01
        num_octave = 3
        octave_scale = 1.4
        iterations = 20

        max_loss = 10.0
        base_image_path = 'C:\\Users\\owatkins\\OneDrive - Analog Devices, Inc\\Documents\\Project Folder\\Tutorials and Courses\\Deep Learning with Python\\European_Landscape.jpg'
        print("Loading Base Image...")

        # Load the base image into Numpy array.
        img = preprocess_image_inception(base_image_path)
        print(f"Image Preprocessed: {img.dtype} of size: {img.shape}")

        # Prepare a list of shape tuples defining the different scales at which to run gradient ascent.
        original_shape = img.shape[1:3]
        successive_shapes = [original_shape]
        for i in range(1, num_octave):
            shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
            successive_shapes.append(shape)

        # Reverse the list so that they run in ascending order.
        successive_shapes = successive_shapes[::-1]

        # Resize the Numpy array of the image to the smallest size.
        original_img = np.copy(img)
        shrunk_original_image = resize_img(original_img, successive_shapes[0])

        # Run deep dream over all octaves.
        for shape in successive_shapes:
            print(f"Processing Image shape: {shape}")

            # Scales up the deep dream image
            img = resize_img(img, shape)

            # Run gradient ascent, altering the dream.
            img = gradient_ascent(img, iterations=iterations, step=step, max_loss=max_loss)

            # Scales up the smaller version of the original image: it will be pixellated. Compute the high-quality
            # version of the original image at this size. The difference between the two is the detail lost in
            # up-scaling.
            upscaled_shrunk_original_img = resize_img(shrunk_original_image, shape)
            same_size_original = resize_img(original_img, shape)
            lost_detail = same_size_original - upscaled_shrunk_original_img

            # Re-inject the lost detail back into the dream. Grab the shrunk_original_image and save the dream at this
            # octave
            img += lost_detail
            shrunk_original_image = resize_img(original_img, shape)
            save_img(img, fname='C:\\Users\\owatkins\\OneDrive - Analog Devices, Inc\\Documents\\Project Folder\\Tutorials and Courses\\Deep Learning with Python\\dream_at_scale_' + str(shape) + '.png')

        # Save the final dream.
        save_img(img, fname='C:\\Users\\owatkins\\OneDrive - Analog Devices, Inc\\Documents\\Project Folder\\Tutorials and Courses\\Deep Learning with Python\\Final_Dream.png')

    def neural_style_transfer():
        """
        Another field of study that arose in the summer of 2015 was the idea of Neural Style Transfer. Style transfer
        involves applying the style of a reference image to a target image while preserving the content of the target
        image. "Style" in the context of an image can mean colours and textures and patterns , whereas the content is
        the higher level macrostructure of the image.

        As with any deep-learning objective, the first job is to define a loss function which we will seek to minimise
        this through training. If we were able to mathematically define "content" and "style", then an appropriate loss
        function to minimise would be:
            loss =   distance(style(reference_image) - style(generated_image))
                   + distance(content(original_image) - content(generated_image))

        For this example we'll use the VGG19 network, given it's a relatively simple pretrained network.

        :return: None
        """
        # Provide the paths to the required images
        target_image_path = 'C:\\Users\\owatkins\\OneDrive - Analog Devices, Inc\\Documents\\Project Folder\\Tutorials and Courses\\Deep Learning with Python\\European_Landscape.jpg'
        style_reference_image_path = 'C:\\Users\\owatkins\\OneDrive - Analog Devices, Inc\\Documents\\Project Folder\\Tutorials and Courses\\Deep Learning with Python\\Landscape_Art_Reference.jpg'

        # Extract the dimensions of the target image, use them to determine the size of the generated image.
        width, height = image.load_img(target_image_path).size
        img_height = 400
        img_width = int(width * img_height / height)

        # Define constants for the reference images and a placeholder for the generated image
        target_image = K.constant(preprocess_image_vgg19(target_image_path, img_h=img_height, img_w=img_width))
        style_reference_image = K.constant(preprocess_image_vgg19(style_reference_image_path, img_h=img_height,
                                                                  img_w=img_width))
        combination_image = K.placeholder((1, img_height, img_width, 3))

        # Combine the 3 images in a single branch. Then load the VGG19 model without the dense classifier, using the
        # input_tensor as the input to the model.
        input_tensor = K.concatenate([target_image, style_reference_image, combination_image], axis=0)
        model = vgg19.VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)
        print("Model Loaded.\n")

        # Now let's define the loss functions for this application.
        def content_loss(base, combination):
            """
            This function computes the L2 distance between the generated image and the content reference image

            :param base: The Content reference image
            :param combination: The generated image
            :return: The L2 distance between the combination image and the base image
            """
            return K.sum(K.square(combination - base))

        def gram_matrix(x):
            """
            Computes the inner product of the correlation matrix of the feature maps of a given layer

            :param x: Activations of a given layer in the network.
            :return: The gram matrix computed on x
            """
            features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
            gram = K.dot(features, K.transpose(features))
            return gram

        def style_loss(style, combination):
            """
            Computes the style loss of the generated image using the gram matrix function defined above.

            :param style:
            :param combination:
            :return:
            """
            S = gram_matrix(style)
            C = gram_matrix(combination)
            channels = 3
            size = img_height * img_width
            return K.sum(K.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))

        def total_variation_loss(x):
            """
            Compute the total variation loss of the input.

            :param x: Input Value
            :return: Variation Loss
            """
            a = K.square(x[:, :img_height - 1, :img_width - 1, :] - x[:, 1:, :img_width-1, :])
            b = K.square(x[:, :img_height - 1, :img_width - 1, :] - x[:, :img_height-1, 1:, :])
            return K.sum(K.pow(a + b, 1.25))
            
        # Dictionary that maps layer names to activation tensors.
        output_dict = dict([(layer.name, layer.output) for layer in model.layers])
        
        # Layers used to measure both the content and style similarities.
        content_layer =   'block5_conv2'
        style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
        
        # Weights applied to each loss style.
        total_variation_weight = 1e-4
        style_weight = 1.0
        content_weight = 0.025
        
        # You'll define the loss by adding all components to this variable.
        loss = K.variable(0.0)
        
        # Add the Content Loss
        layer_features = output_dict[content_layer]
        target_image_features = layer_features[0, :, :, :]
        combination_features = layer_features[2, :, :, :]
        loss = loss + (content_weight * content_loss(target_image_features, combination_features))
        
        # Add the Style Loss
        for layer_name in style_layers:
            layer_features = output_dict[layer_name]
            style_reference_features = layer_features[1, :, :, :]
            combination_features = layer_features[2, :, :, :]
            s1 = style_loss(style_reference_features, combination_features)
            loss = loss + ((style_weight / len(style_layers)) * s1)
            
        # Add the total variation loss
        loss = loss + (total_variation_weight * total_variation_loss(combination_image))
        
        # Now let's setup the gradient descent process. The particular process we want to use comes packaged in SciPy,
        # however due to the limitations of the packaged process we'll define our own class called Extractor to handle
        # computing the loss and gradient values without running redundant computations.
        grads = K.gradients(loss, combination_image)[0]
        fetch_loss_and_grads = K.function([combination_image], [loss, grads])
        
        class Evaluator(object):
            
            def __init__(self):
                self.loss_value = None
                self.grads_values = None
                
            def loss(self, x):
                assert self.loss_value is None
                x = x.reshape((1, img_height, img_width, 3))
                outs = fetch_loss_and_grads([x])
                loss_value = outs[0]
                grads_value = outs[1].flatten().astype('float64')
                self.loss_value = loss_value
                self.grads_values = grads_value
                return self.loss_value
                
            def grads(self, x):
                assert self.loss_value is not None
                grads_values = np.copy(self.grads_values)
                self.loss_value = None
                self.grads_values = None
                return grads_values
                
        evaluator = Evaluator()
        
        # Now let's attempt training
        result_prefix = 'C:\\Users\\owatkins\\OneDrive - Analog Devices, Inc\\Documents\\Project Folder\\Tutorials and Courses\\Deep Learning with Python\\my_result'
        iterations = 20
        
        x = preprocess_image_vgg19(target_image_path, img_height, img_width)
        x = x.flatten()
        
        for i in range(iterations):
            print(f"Start of iteration {i}")
            start_time = time.time()
            x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x, fprime=evaluator.grads, maxfun=20)
            print(f"Current Loss value: {min_val}")
            img = x.copy().reshape((img_height, img_width, 3))
            img = deprocess_image_vgg19(img)
            fname = result_prefix + '_at_iteration_%d.png' % i
            save_img(img, fname, False)
            print(f"Image saved as: {fname}")
            end_time = time.time()
            print(f"Iteration {i} completed in {end_time - start_time}s")
            
    def variational_auto_encoder():
        """
        This function will detail an example of creating, training and using a Variational Autoencoder (VAE).
        For the sake of easy programming and speedy computations we'll use the MNIST dataset in this example.
        
        :return: None
        """
        
        # Prepare the input layers of the Encoder to handle MNIST dataset.
        img_shape = (28, 28, 1)
        batch_size = 16
        latent_dim = 2
        
        input_img = keras.Input(shape=img_shape)
        
        # Now let's provide a collection of layers to handle encoding the image input.
        x = layers.Conv2D(32, 3, padding='same', activation='relu')(input_img)
        x = layers.Conv2D(64, 3, padding='same', activation='relu', strides=(2, 2))(x)
        x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
        x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
        
        # Finally let's encode the image input using Dense layers. Be sure to grab the shape of the Tensor
        # before flattening for use later.
        shape_before_flattening = K.int_shape(x)
        
        x = layers.Flatten()(x)
        z_mean = layers.Dense(latent_dim)(x)
        z_log_var = layers.Dense(latent_dim)(x)
        
        def sampling(args):
            """
            This function will encode and sample the encoding space. We'll use this function in a Lambda 
            layer later.
            
            :param: args:
            :return: A sampled point in the encoded space.
            """
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.0, stddev=1.0)
            
            return z_mean + K.exp(z_log_var) * epsilon
        
        # Use the sampling function in a Lambda layer.
        z = layers.Lambda(sampling)([z_mean, z_log_var])
        
        # Now let's make the decoder.
        decoder_input = layers.Input(K.int_shape(z)[1:])
        
        # Upsanple the image.
        x = layers.Dense(np.prod(shape_before_flattening[1:]), 
                         activation='relu')(decoder_input)
                         
        # Reshape z to match the shape of z prior to flattening.
        x = layers.Reshape(shape_before_flattening[1:])(x)
                         
        # Use a Conv2DTranspose and a Conv2D to decode z into an output that's the same size as the input.
        x = layers.Conv2DTranspose(32, 3, padding='same', activation='relu', strides=(2, 2))(x)
        x = layers.Conv2D(1, 3, padding='same', activation='sigmoid')(x)
        
        # Build the decoder model and define the decoder output.
        decoder = models.Model(decoder_input, x)
        z_decoded = decoder(z)
        
        # The dual loss of a VAE doesn't fit the tradional expectation of a sample-wise function of the normal format
        # loss(input, target). Thus we'll need to setup the loss by writing a custom layer that internally uses a 
        # built-in add_loss layer method to create an arbitrary loss.
        class CustomVariationLayer(keras.layers.Layer):
        
            def vae_loss(self, x, z_decoded):
                """
                Custom loss function for VAE's
                
                :param: self:
                :param: x:
                :param: z_decoded:
                :return:
                """
                x = K.flatten(x)
                z_decoded = K.flatten(z_decoded)
                xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
                kl_loss = -5e-4 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
                return K.mean(xent_loss + kl_loss)
                
            def call(self, inputs, **kwargs):
                """
                This function is used to implement the custom VAE loss function defined above. The output from the
                function is not needed by the caller, however the function is required to return something.

                :param: self: CustomVariationLayer object
                :param: inputs: A list containing both the input values and the decoded values.
                :return: Anything, the function output isn't used by the caller.
                """
                x = inputs[0]
                z_decoded = inputs[1]
                loss = self.vae_loss(x, z_decoded)
                self.add_loss(loss, inputs=inputs)
                return x

        y = CustomVariationLayer()([input_img, z_decoded])

        # Now let's compile and train the model.
        vae = models.Model(input_img, y)
        vae.compile(optimizer='rmsprop', loss=None)
        vae.summary()

        (x_train, _), (x_test, y_test) = mnist.load_data()

        x_train = x_train.astype('float32') / 255.0
        x_train = x_train.reshape(x_train.shape + (1,))
        x_test = x_test.astype('float32') / 255.0
        x_test = x_test.reshape(x_test.shape + (1,))

        vae.fit(x=x_train, y=None, shuffle=True, epochs=10, batch_size=batch_size, validation_data=(x_test, None))

        # Display a grid of 15 x 15 digits (255 digits total).
        n = 15
        digit_size = 28
        figure = np.zeros((digit_size * n, digit_size * n))

        # Transforms linearly spaced coordinates using SciPy ppf function to produce values of the latent variable z
        # (because the prior of the latent space is Gaussian)
        grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
        grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                # Repeats z multiple times to form a complete batch
                z_sample = np.array([[xi, yi]])
                z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)

                # decodes the batch in digit images
                x_decoded = decoder.predict(z_sample, batch_size=batch_size)
                # Reshapes the first digit in the batch from 28 x 28 x 1 to 28 x 28
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size, j * digit_size: (j + 1) * digit_size] = digit

        # Plot the figure in shades of Grey.
        plt.figure(figsize=(10, 10))
        plt.imshow(figure, cmap='Greys_r')
        plt.show()

    def generative_adversarial_networks():
        """
        This section details the design, training and deployment of Generative Adversarial Networks (GAN's).
        When building a GAN it is a good idea to search for design hints and tips, such as those provided in this
        chapter. More exhaustive searches will result in more useful tips, but designing GAN's is notoriously difficult.

        We'll use the CIFAR10 dataset for this example.

        :return: None
        """
        # Let's start by building the Generator model.
        latent_dim = 32
        height = 32
        width = 32
        channels = 3

        generator_input = keras.Input(shape=(latent_dim,))

        # Transform the input to a 16 x 16 28 channel feature map.
        x = layers.Dense(128 * 16 * 16)(generator_input)
        x = layers.LeakyReLU()(x)
        x = layers.Reshape((16, 16, 128))(x)

        x = layers.Conv2D(256, 5, padding='same')(x)
        x = layers.LeakyReLU()(x)

        # Upsample to 32 x 32.
        x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2D(256, 5, padding='same')(x)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2D(256, 5, padding='same')(x)
        x = layers.LeakyReLU()(x)

        # Produce a 32 x 32 1-channel feature map, same shape as the CIFAR10 dataset
        x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)

        # Now instantiate the generator model, which maps a vector of shape (latent_dm,) to an image of shape
        # (32 x 32 x 3)
        generator = models.Model(generator_input, x)
        generator.summary()

        # Now let's build a discriminator (adversary) network.
        discriminator_input = keras.Input(shape=(height, width, channels))
        x = layers.Conv2D(128, 3)(discriminator_input)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2D(128, 4, strides=2)(x)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2D(128, 4, strides=2)(x)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2D(128, 4, strides=2)(x)
        x = layers.LeakyReLU()(x)
        x = layers.Flatten()(x)

        # One dropout layer, very important trick. Then add a classification layer.
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(1, activation='sigmoid')(x)

        # Instantiate the discriminator model, which takes a 32 x 32 x 3 image and provides a binary score: Real/Fake.
        discriminator = models.Model(discriminator_input, x)
        discriminator.summary()

        # Provide an RMSprop optimizer with gradient clipping (by value) and learning-rate decay for stability.
        # Then compile the model.
        discriminator_optimizer = keras.optimizers.RMSprop(lr=0.0008, clipvalue=1.0, decay=1e-8)
        discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')

        # Be very careful when training a GAN. The goal is to have a generator which is good enough to get the
        # discriminator to predict 'real' for a fake image. To achieve this, while training the GAN the discriminator
        # cannot be allowed to train. If this is not done you'd end up training the discriminator to ALWAYS predict
        # 'real', which helps no one.
        discriminator.trainable = False

        gan_input = keras.Input(shape=(latent_dim,))
        gan_output = discriminator(generator(gan_input))
        gan = models.Model(gan_input, gan_output)

        gan_optimizer = keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8)
        gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')

        # How to train a GAN:
        #   1) Draw random points in the latent space (random noise).
        #   2) Generate images with the generator using this random noise.
        #   3) Mix the generated images with real ones.
        #   4) Train discriminator using these mixed images, with corresponding targets: either "real" (dataset images)
        #      or "fake" (generated images).
        #   5) Draw new random points in the latent space.
        #   6) Train gan using these random vectors, with targets that all say "these are real images". This updates the
        #      weights of the generator (only, because the discriminator is frozen inside gan) to move toward getting
        #      the discriminator to predict "these images are real" for generated images, ie: this trains the generator
        #      to fool the discriminator.

        # Now let's load the cifar10 dataset and only use the "frog" class.
        (x_train, y_train), (_, _) = cifar10.load_data()
        x_train = x_train[y_train.flatten() == 6]

        # Normalise data
        x_train = x_train.reshape((x_train.shape[0],) + (height, width, channels)).astype('float32') / 255.0

        # The save_dir is where samples of generated images will be saved to.
        iterations = 10000
        batch_size = 20
        save_dir = 'C:\\Users\\owatkins\\OneDrive - Analog Devices, Inc\\Documents\\Project Folder\\' \
                   'Tutorials and Courses\\Deep Learning with Python\\gan_images'

        start = 0
        for step in range(iterations):
            # Sample random points in the latent space.
            random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

            # Generate some images.
            generated_images = generator.predict(random_latent_vectors)

            # Concatenate generated and real images into one batch, as well as create a training target tensor called
            # labels
            stop = start + batch_size
            real_images = x_train[start: stop]
            combined_images = np.concatenate([generated_images, real_images])

            labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
            labels += 0.05 * np.random.random(labels.shape)

            # Train the discriminator first, only on a single batch.
            d_loss = discriminator.train_on_batch(combined_images, labels)

            # Sample the latent space again. This time label each images as real.
            random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
            misleading_targets = np.zeros((batch_size, 1))

            # Train the generator only to make better images.
            a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)

            start += batch_size
            if start > len(x_train) - batch_size:
                start = 0

            # Every 100 steps save the network weights and some sample images for reference.
            if step % 100 == 0:
                gan.save_weights('C:\\Users\\owatkins\\OneDrive - Analog Devices, Inc\\Documents\\Project Folder\\'
                                 'Tutorials and Courses\\Deep Learning with Python\\gan_progress\\gan_' + str(step) + '.h5')

                print(f"discriminator loss: {d_loss}")
                print(f"adversarial loss: {a_loss}")

                img = image.array_to_img(generated_images[0] * 255.0, scale=False)
                img.save(os.path.join(save_dir, 'generated_frog_' + str(step) + '.png'))

                img = image.array_to_img(real_images[0] * 255.0, scale=False)
                img.save(os.path.join(save_dir, 'real_frog_' + str(step) + '.png'))

        # Save the final model.
        gan.save('C:\\Users\\owatkins\\OneDrive - Analog Devices, Inc\\Documents\\Project Folder\\'
                 'Tutorials and Courses\\Deep Learning with Python\\gan_progress\\gan_final.h5')

    # text_generation()
    # deep_dream()
    # neural_style_transfer()
    # variational_auto_encoder()
    generative_adversarial_networks()
