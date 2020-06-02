"""
Chapter 7: Advanced Deep-Learning Best Practices

This chapter covers:
    -> The Keras Functional API
    -> Using Keras callbacks
    -> Working with the TensorBoard visualisation tool
    -> Important best practices for developing state-of-the-art models
"""

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

    # functional_api_eg()
