import tensorflow as tf

def get_solvers(learning_rate=1e-3, beta1=0.5):
    D_solver = tf.optimizers.Adam(learning_rate, beta1)
    G_solver = tf.optimizers.Adam(learning_rate, beta1)
    return D_solver, G_solver
