import tensorflow as tf

def generator(noise_dim=96):
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(noise_dim,)),
        tf.keras.layers.Dense(1024),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(1024),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(784,activation=tf.nn.tanh)
    ])
    return model
