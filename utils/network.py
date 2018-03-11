import tensorflow as tf
import numpy as np

def build_mlp(
        mlp_input,
        output_size,
        scope,
        n_layers,
        size,
        output_activation=None):
    '''
    Build a feed forward network (multi-layer-perceptron, or mlp)
    with 'n_layers' hidden layers, each of size 'size' units.
    Use tf.nn.relu nonlinearity between layers.
    Args:
            mlp_input: the input to the multi-layer perceptron
            output_size: the output layer size
            scope: the scope of the neural network
            n_layers: the number of layers of the network
            size: the size of each layer:
            output_activation: the activation of output layer
    Returns:
            The tensor output of the network

    TODO: Implement this function. This will be similar to the linear
    model you implemented for Assignment 2.
    "tf.layers.dense" or "tf.contrib.layers.fully_connected" may be helpful.

    A network with n layers has n
      linear transform + nonlinearity
    operations before a final linear transform for the output layer
    (followed by the output activation, if it is not None).

    '''
    #######################################################
    #########   YOUR CODE HERE - 7-20 lines.   ############
    h = mlp_input
    out = mlp_input

    with tf.variable_scope(scope):
        n = tf.layers.flatten(mlp_input)
        for i in range(n_layers):
            h_scope = "h" + str(i)
            n = tf.contrib.layers.fully_connected(n, num_outputs=size, activation_fn=tf.nn.relu, reuse=tf.AUTO_REUSE, scope=h_scope)
        out = tf.contrib.layers.fully_connected(n, num_outputs=output_size, activation_fn=output_activation, reuse=tf.AUTO_REUSE, scope="out")

    return out

        #######################################################
    #########          END YOUR CODE.          ############


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def entropy(logits):
    a0 = logits - tf.reduce_max(logits, axis=1, keepdims=True)
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, axis=1, keepdims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=1)