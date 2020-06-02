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
from tensorflow.keras import Input

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

    # functional_api_eg()
    # multi_input_model_eg()
    # multi_output_model_eg()
