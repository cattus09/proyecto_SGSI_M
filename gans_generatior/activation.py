import tensorflow as tf

def leaky_relu(x, alpha=0.01):
    x = tf.nn.leaky_relu(x,alpha)
    return x