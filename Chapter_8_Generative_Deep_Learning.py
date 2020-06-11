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
from tensorflow import keras
from tensorflow.keras import models, layers

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
        print(f"Corpus Length: {len(text)}")

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
            print(f"{epoch} Epoch")
            # Train for 1 epoch
            model.fit(x, y, batch_size=128, epochs=1)

            # provide the seed information for the model's prediction
            start_index = np.random.randint(0, len(text) - maxlen - 1)
            generated_text = text[start_index: start_index + maxlen]
            print('--- Generating with seed: "' + generated_text + '"')

            # Now examine the effect of the temperature factor
            for temperature in [0.2, 0.5, 1.0, 1.2]:
                # Print the current temperature value and the seed text.
                print('------ temperature: ', temperature)
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

    text_generation()
