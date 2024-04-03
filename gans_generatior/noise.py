import tensorflow as tf
import numpy as np

def sample_noise(batch_size, dim):
    noise = tf.random.uniform([batch_size,dim],minval=-1,maxval=1)
    return noise