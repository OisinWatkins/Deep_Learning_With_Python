"""
Chapter 7: Advanced Deep-Learning Best Practices

This chapter covers:
    -> The Keras Functional API
    -> Using Keras callbacks
    -> Working with the TensorBoard visualisation tool
    -> Important best practices for developing state-of-the-art models
"""
import numpy as np
from tensorflow.keras import models, layers
from tensorflow.keras import Input, applications, callbacks
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

if __name__ == '__main__':
    """
    Run using `python -m Chapter_7_Advanced_Deep_Learning_Best_Practices`
    """

    def functional_api_eg():
        """
        This function will simply define 2 models, 1 using the Sequential() architecture and one using the keras
        functional API

        :return: None
        """

        # Let's first make a simple densely connected network with a softmax output
        seq_model = models.Sequential()
        seq_model.add(layers.Dense(32, activation='relu', input_shape=(64,)))
        seq_model.add(layers.Dense(32, activation='relu'))
        seq_model.add(layers.Dense(10, activation='softmax'))

        seq_model.summary()

        # Now let's make the same model using the functional API
        input_tensor = Input(shape=(64,))
        x = layers.Dense(32, activation='relu')(input_tensor)
        x = layers.Dense(32, activation='relu')(x)
        output_tensor = layers.Dense(10, activation='softmax')(x)

        model = models.Model(input_tensor, output_tensor)
        model.summary()

    def multi_input_model_eg():
        """
        This function will detail the building and use of a multi-input model using a simple question-answering problem

        :return: None
        """
        text_vocabulary_size = 10000
        question_vocabulary_size = 10000
        answer_vocabulary_size = 500

        # Create a text input tensor of variable length. Note that you can optionally name inputs
        text_input = Input(shape=(None,), dtype='int32', name='text')

        # Now embed and encode the text input through a sequence vector of size of 64 into a single vector
        embedded_text = layers.Embedding(64, text_vocabulary_size)(text_input)
        encoded_text = layers.LSTM(32)(embedded_text)

        # Create a question input tensor of variable length. Note that you can optionally name inputs
        question_input = Input(shape=(None,), dtype='int32', name='question')

        # Now embed and encode the text input through a sequence vector of size of 64 into a single vector
        embedded_question = layers.Embedding(64, question_vocabulary_size)(question_input)
        encoded_question = layers.LSTM(16)(embedded_question)

        # Concatenate the encoded questions and text
        concatenated = layers.concatenate([encoded_text, encoded_question], axis=-1)

        # Add a softmax classifier to the top
        answer = layers.Dense(answer_vocabulary_size, activation='softmax')(concatenated)

        # Now define the model, providing the 2 inputs and one output
        model = models.Model([text_input, question_input], answer)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
        model.summary()

        # Now make some random inputs and outputs to train with for the sake of this example
        num_samples = 1000
        max_len = 100

        text = np.random.randint(1, text_vocabulary_size, size=(num_samples, max_len))
        question = np.random.randint(1, question_vocabulary_size, size=(num_samples, max_len))
        answers = np.random.randint(0, 1, size=(num_samples, answer_vocabulary_size))

        # Note that it is possible to train using either of the below code segments
        # model.fit([text, question], answers, epochs=10, batch_size=128)
        model.fit({'text': text, 'question': question}, answers, epochs=10, batch_size=128)

    def multi_output_model_eg():
        """
        This function will detail the building and use of a multi-output model using a simple social-media example

        :return: None
        """

        # Dummy variables to make sure there are no errors
        posts = None
        age_targets = None
        income_targets = None
        gender_targets = None
        vocabulary_size = 50000
        num_income_groups = 10

        # Make an input to hold the posts shared by users, variable length just like in the previous example
        post_input = Input(shape=(None,), dtype='int32', name='posts')

        # Embed and encode these inputs. This time we'll use a larger Conv1D network
        embedded_posts = layers.Embedding(256, vocabulary_size)(post_input)
        x = layers.Conv1D(128, 5, activation='relu')(embedded_posts)
        x = layers.MaxPooling1D(5)(x)
        x = layers.Conv1D(256, 5, activation='relu')(x)
        x = layers.Conv1D(256, 5, activation='relu')(x)
        x = layers.MaxPooling1D(5)(x)
        x = layers.Conv1D(256, 5, activation='relu')(x)
        x = layers.Conv1D(256, 5, activation='relu')(x)
        x = layers.GlobalMaxPool1D()(x)
        x = layers.Dense(128, activation='relu')(x)

        # Now make 3 separate heads, one for each output. Note how all 3 have unique activations. This will require
        # unique loss functions for each head.
        age_prediction = layers.Dense(1, name='age')(x)
        income_prediction = layers.Dense(num_income_groups, activation='softmax', name='income')(x)
        gender_prediction = layers.Dense(1, activation='sigmoid', name='gender')(x)

        # Instantiate the model
        model = models.Model(post_input, [age_prediction, income_prediction, gender_prediction])

        # Compile and fit the model. Use the dictionary method of passing parameters for greater explicitness. Note that
        # very imbalanced loss contributions will lead to the model being more optimised for the output with the largest
        # individual loss than for the others. To counteract this we can scale the individual loss functions to diminish
        # the largest contributor (mse ~= 3-5) and magnify the smallest contributor (binary_crossentropy ~= 0.1)
        model.compile(optimizer='rmsprop',
                      loss={'age': 'mse',
                            'income': 'categorical_crossentropy',
                            'gender': 'binary_crossentropy'},
                      loss_weights={'age': 0.25,
                                    'income': 1.0,
                                    'gender': 10.0})
        model.fit(posts, {'age': age_targets,
                          'income': income_targets,
                          'gender': gender_targets},
                  epochs=10, batch_size=64)

    def inception_eg():
        """
        This function will use the keras functional API to build a simple Inception module. Inception modules come
        packaged as standard with Keras, however this function will detail how they can be built if a more custom module
        is needed

        :return: None
        """

        # placeholder input, not really necessary for this example
        x = Input(shape=(None,), dtype=float)

        # Create the 1st branch which performs a simple Conv2D operation with a kernel 1x1 in size
        branch_a = layers.Conv2D(128, 1, activation='relu', strides=2)(x)

        # Create the 2nd branch which performs a simple Conv2D operation with a kernel 1x1 in size, and then a
        # subsequent Conv2D with both a larger kernel and a larger step-size
        branch_b = layers.Conv2D(128, 1, activation='relu')(x)
        branch_b = layers.Conv2D(128, 3, activation='relu', strides=2)(branch_b)

        # Create the 3rd branch which performs an average pooling operation first, and then a
        # subsequent Conv2D operation.
        branch_c = layers.AveragePooling2D(3, strides=2)(x)
        branch_c = layers.Conv2D(128, 3, activation='relu', strides=2)(branch_c)

        # Create the 4th branch, which is the deepest of all 4.
        branch_d = layers.Conv2D(128, 1, activation='relu')(x)
        branch_d = layers.Conv2D(128, 3, activation='relu')(branch_d)
        branch_d = layers.Conv2D(128, 3, activation='relu', strides=2)(branch_d)

        # Concatenate the outputs from each branch to create the output for the whole module.
        output = layers.concatenate([branch_a, branch_b, branch_c, branch_d], axis=-1)

    def residual_eg():
        """
        The function will use the functional API to demonstrate how residual connections can be made in an acyclic graph

        :return: None
        """

        # placeholder input tensor
        x = Input(shape=(None,), dtype=float)

        # Example using simple add layer
        y = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        y = layers.Conv2D(128, 3, activation='relu', padding='same')(y)
        y = layers.Conv2D(128, 3, activation='relu', padding='same')(y)

        y = layers.add([y, x])

        # Example using a separate residual branch with its own transformation
        y = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        y = layers.Conv2D(128, 3, activation='relu', padding='same')(y)
        y = layers.MaxPooling2D(2, strides=2)(y)

        residual = y = layers.Conv2D(128, 1, strides=2, padding='same')(x)
        y = layers.add([y, residual])

    def layer_weight_sharing():
        """
        In the example provided we'll make a simple network designed to compare the semantic relationship between 2
        sentences and provide a score between 0 (totally different) and 1 (identical in meaning). Here we don't want to
        have 2 different that learn independent transformations. It would instead be more useful to have 1 layer be
        reused, learning only one set of transformations to be used on both input sentences (a.k.a. a Siamese Model)

        :return: None
        """
        left_data = []
        right_data = []
        targets = []

        # Single LSTM layer which will be used in bith data paths
        lstm = layers.LSTM(32)

        # Left data lane
        left_input = Input(shape=(None, 128))
        left_output = lstm(left_input)

        # Right data lane
        right_input = Input(shape=(None, 128))
        right_output = lstm(right_input)

        # Merge outputs and create predictions
        merged = layers.concatenate([left_output, right_output], axis=-1)
        predictions = layers.Dense(1, activation='sigmoid')(merged)

        model = models.Model([left_input, right_input], predictions)
        model.fit([left_data, right_data], targets)

    def models_as_layers():
        """
        Similar to the previous example, we can use entire models an arbitrary number of times in a single model. This
        can be quite useful in certain applications, for example here we use a Siamese model to process 2 camera inputs
        using pre-learned representations. From there we could add a dense head to also calculate the distance to any
        detected object

        :return: None
        """
        xception_base = applications.Xception(weights=None, include_top=False)

        left_input = Input(shape=(255, 255, 3))
        right_input = Input(shape=(255, 255, 3))

        left_features = xception_base(left_input)
        right_features = xception_base(right_input)

        merged_features = layers.concatenate([left_features, right_features], axis=-1)

        # Now build the rest of the model.

    def examples_of_training_callbacks():
        """
        This function will go into detail on how to modify the callback settings for a training cycle. The callbacks
        supported by Keras include, but are not limited to the following:
            -> keras.callbacks.ModelCheckpoint
            -> keras.callbacks.EarlyStopping
            -> keras.callbacks.LearningRateScheduler
            -> keras.callbacks.ReduceLROnPlateau
            -> keras.callbacks.CSVLogger
        As examples, we'll now detail the use of EarlyStopping and ReduceLROnPlateau

        :return: None
        """

        # Make some mock training data and a simple dense model for the sake of this example
        x_train = []
        y_train = []
        tmp_model = models.Sequential()
        tmp_model.add(layers.Dense(1, activation='sigmoid', input_shape=(256, 256)))

        # The first callback list is designed to interrupt the training if the training accuracy stops improving for
        # more that 1 epochs, and to save the model any time the validation loss improves (the model will only be
        # overwritten if the validation loss improves)
        callbacks_list = [callbacks.EarlyStopping(monitor='acc', patience=1),
                          callbacks.ModelCheckpoint(filepath='my_model.h5', monitor='loss_val', save_best_only='True')]

        # Now compile and train the network. The callbacks monitor accuracy, so you should do so as well. Given that the
        # callbacks also monitor validation loss the training must incorporate validation.
        tmp_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
        tmp_model.fit(x_train, y_train, epochs=10, callbacks=callbacks_list, validation_split=0.2)

        # This callback list is designed to dynamically change the learning rate if the validation loss stops improving
        # for 10 epochs. Note that the factor value will accumulate for every count on the towards the patience limit
        # (i.e: after 10 epochs of no improvement the LR will be divided by 10)
        callbacks_list = [callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)]

        # Compile and train the model. Note again that the callback monitors validation loss so validation must occur
        # during training.
        tmp_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
        tmp_model.fit(x_train, y_train, epochs=10, callbacks=callbacks_list, validation_split=0.2)

        # Now let's try writing a custom callback. Building a subclass is supported by transparently named methods,
        # including: "on_epoch_begin", "on_epoch_end", "on_batch_begin", "on_batch_end", "on_train_being",
        # "on_train_end".
        class ActivationLogger(callbacks.Callback):
            """
            This callback is designed to save the activations of each layer in the model when provided a validation
            sample into a .npz file at the end of every epoch.
            """
            def set_model(self, model):
                """
                This function creates a new model using the activations of each layer in the model provided as the
                outputs.

                :param model: keras Model being trained
                :return: None
                """
                self.model = model
                layer_outputs = [layer.output for layer in model.layers]
                self.activations_model = models.Model(model.input, layer_outputs)

            def on_epoch_end(self, epoch, logs=None):
                """
                This functions writes the activation outputs from the activations model when given a validation sample
                into a unique .npz file.

                :param epoch: Epoch Number
                :param logs: Event logs
                :return: None
                """
                if self.validation_data is None:
                    raise RuntimeError('Requires validation_data.')
                validation_sample = self.validation_data[0][0:1]
                activations = self.activations_model.predict(validation_sample)
                f = open('activations_at_epoch_' + str(epoch) + '.npz', 'w')
                np.savez(f, activations)
                f.close()

    def tensorboard_eg():
        """
        To do good research or to build good models, you need rich, frequent feedback about what's going on inside your
        model during experiments. Keras is designed to bring a developer from "Idea" to "Experiment" in as short a time
        as possible. Tensorboard is a browser-based data visualisation tool designed with neural network training in
        mind. For example, Tensorboard allows you to:
            -> Visually monitor metrics during training
            -> Visualise your model architecture
            -> Visualise histograms of activations and gradients
            -> Explore embeddings in 3D

        Let's demonstrate Tensorboard's usefulness by training a simple 1D convnet on the IMDB dataset. (Experiencing a
        lot of compatibility issues) https://www.tensorflow.org/tensorboard/get_started

        :return: None
        """

        max_features = 2000
        max_len = 500

        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
        x_train = sequence.pad_sequences(x_train, maxlen=max_len)
        x_test = sequence.pad_sequences(x_test, maxlen=max_len)

        model = models.Sequential()
        model.add(layers.Embedding(max_features, 128, input_length=max_len, name='embed'))
        model.add(layers.Conv1D(32, 7, activation='relu'))
        model.add(layers.MaxPooling1D(5))
        model.add(layers.Conv1D(32, 7, activation='relu'))
        model.add(layers.GlobalMaxPool1D())
        model.add(layers.Dense(1))

        model.summary()
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

        callbacks_list = [callbacks.TensorBoard(log_dir="C:\\Datasets\\IMDB\\my_log_dir",
                                                histogram_freq=1)]  # embeddings_freq=1
        history = model.fit(x_train, y_train, epochs=20, batch_size=128, validation_split=0.2, callbacks=callbacks_list)

    def getting_the_most_from_your_model():
        """
        This function will go into techniques that will pull your models from a "works okay" level to "wins ML
        competitions" level. Each technique will be given a few lines of code and some comments to describe how the work
        and where they're useful.

        :return: None
        """

        # Technique No.1: Batch Normalisation
        #   Normalisation is a broad category of methods that seek to make different samples seen by a model more
        #   similar to each other, which help the model generalise well to new data. A very common form of normailsation
        #   is one we've used many times in this project so far: Subtracting the mean and dividing by the standard
        #   deviation. This method assumes the data has a Guassian distribution, and yields data which is zero-centered
        #   with unit variance. Do be aware of the axis parameter in the BatchNormailzation layer, it defaults to -1,
        #   which is the correct value for almost every layer type, however this may change for custom layers.
        data = np.zeros(shape=(1, 1))
        normailsed_data = (data - np.mean(data, axis=...)) / np.std(data, axis=...)

        # So far batch normailsation has only been used on input data, however for very deep models normalisation is
        # helpful/required for avoiding vanishing gradient and ensuring that no transformation pushes the data into a
        # non-zero-centered batch with variance > 1
        conv_model = models.Sequential()
        conv_model.add(layers.Conv2D(32, 3, activation='relu'))
        conv_model.add(layers.BatchNormalization())

        dense_model = models.Sequential()
        dense_model.add(layers.Conv2D(32, 3, activation='relu'))
        dense_model.add(layers.BatchNormalization())

        # A more experimental method that seems to show a lot of promise is the use of seperable Conv2D layers in place
        # of regular Conv2D layers. These layers have shown that they result in faster, lighter-weight models that learn
        # better representations due to their more data efficient transformation.
        sep_model = models.Sequential()

        sep_model.add(layers.SeparableConv2D(32, 3, activation='relu', input_shape=(64, 64, 3)))
        sep_model.add(layers.SeparableConv2D(64, 3, activation='relu'))
        sep_model.add(layers.MaxPooling2D(2))

        sep_model.add(layers.SeparableConv2D(64, 3, activation='relu', input_shape=(64, 64, 3)))
        sep_model.add(layers.SeparableConv2D(128, 3, activation='relu'))
        sep_model.add(layers.MaxPooling2D(2))

        sep_model.add(layers.SeparableConv2D(64, 3, activation='relu', input_shape=(64, 64, 3)))
        sep_model.add(layers.SeparableConv2D(128, 3, activation='relu'))
        sep_model.add(layers.GlobalAveragePooling2D())

        sep_model.add(layers.Dense(32, activation='relu'))
        sep_model.add(layers.Dense(10, activation='softmax'))

        sep_model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        sep_model.summary()

        # Hyperparameter tuning is a very powerful technique for better results with your model, however there are no
        # hard and fast rules, or even equations that can really solve the problem, given that essentially no output
        # metric has a direct or differentiable relationship to the hyperparameters. Normally people use intuition and
        # guess repeatedly at the correct hyperparameters and this usually works out well. There are tools to help:
        #   -> https://github.com/hyperopt/hyperopt
        #   -> https://github.com/maxpumperla/hyperas
        # Be careful with these tools, overfitting to the validation data is always likely.

        # Another powerful tool is model ensembling, where we use multiple models to make their own independent
        # predictions on the input data and then combine these predictions before making the final prediction. This is
        # nearly always the method that creates competition-winning models, although the fine details of each ensemble
        # must either be learned or guessed at empirically. Diversity is key for this technique, the more different each
        # model is, and the better each model is on its own, the better the end result tends to be.
        model_a = models.Sequential()
        model_b = models.Sequential()
        model_c = models.Sequential()
        model_d = models.Sequential()

        preds_a = model_a.predict(x=np.zeros(shape=(1, 1)))
        preds_b = model_a.predict(x=np.zeros(shape=(1, 1)))
        preds_c = model_a.predict(x=np.zeros(shape=(1, 1)))
        preds_d = model_a.predict(x=np.zeros(shape=(1, 1)))

        # All weight values are assumed to be learned empirically.
        final_pred = 0.25 * (preds_a + preds_b + preds_c + preds_d)
        final_pred = (0.5 * preds_a) + (0.25 * preds_b) + (0.1 * preds_c) + (0.15 * preds_d)

    # functional_api_eg()
    # multi_input_model_eg()
    # multi_output_model_eg()
    # inception_eg()
    # residual_eg()
    # layer_weight_sharing()
    # models_as_layers()
    # examples_of_training_callbacks()
    # tensorboard_eg()
    # getting_the_most_from_your_model()
