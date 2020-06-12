"""
Chapter 8: Generative Deep Learning

This chapter covers:
    -> Text Generation with LSTM
    -> Implementing DeepDream
    -> Performing Natural Style Transfer
    -> Variational Autoencoders
    -> Understanding Generative Adversarial Networks
"""
import sys
import numpy as np
from PIL import Image
from matplotlib import cm
from tensorflow import keras
from tensorflow.keras import models, layers
from tensorflow.keras.applications import inception_v3
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
    We're going to define a bunch of auxiliary functions before using deep_dream()
    """
    def resize_img(img, size):
        """
        Handles resizing the image using Pillow, deprocess_image and numpy.

        :param img: Image
        :param size: Desired Size
        :return: The input image resized to the desired size as np array.
        """
        pil_img = deprocess_image(img)
        factors = (float(size[0]) / img.shape[1],
                   float(size[1]) / img.shape[2])
        return np.array(img.resize(factors))

    def save_img(img, fname):
        """
        Saves the image using the Pillos.Image library and the deprocess_image function.

        :param img: Image as numpy array
        :param fname: Filename for the new image
        :return: None
        """
        pil_image = deprocess_image(np.copy(img))
        pil_image.save(fname)

    def preprocess_image(image_path):
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

    def deprocess_image(x):
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
            loss += coeff * K.sum(K.square(activation[:, 2: -2, 2: -2, :])) / scaling

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
        base_image_path = '/European_Landscape.jpg'

        # Load the base image into Numpy array.
        img = preprocess_image(base_image_path)

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
            save_img(img, fname='/dream_at_scale_' + str(shape) + '.png')

        # Save the final dream.
        save_img(img, fname='/Final_Dream.png')


    # text_generation()
    deep_dream()
