
import tensorflow as tf
from src.modules.module import Module


def weights_pretrained(weights, name=''):
    return tf.get_variable(name+'/weights', initializer=weights)


def biases_pretrained(bias, name=''):
    return tf.get_variable(name+'/biases', initializer=bias)


def fprop_next_linear(x, f, w, b, alpha):
    beta = alpha - 1
    v_max = tf.nn.relu(w)
    v_min = tf.minimum(0.0, w)
    z_max = tf.matmul(x, v_max) + tf.maximum(b, 0)
    z_min = tf.matmul(x, v_min) + tf.maximum(b, 0)
    s_max = alpha * tf.divide(f, z_max)
    s_min = -beta * tf.divide(f, z_min)
    c = tf.matmul(s_max, tf.transpose(v_max)) + tf.matmul(s_min, tf.transpose(v_min))
    return x * c


def fprop_first_linear(x, f, w, b, lowest, highest):
    v = tf.nn.relu(w)
    u = tf.minimum(w, 0.0)
    l = x * 0 + lowest
    h = x * 0 + highest
    z = tf.matmul(x, w) - tf.matmul(l, v) - tf.matmul(h, u) + tf.maximum(b, 0)
    s = tf.divide(f, z)
    f = x * tf.matmul(s, tf.transpose(w)) - l * tf.matmul(s, tf.transpose(v)) - h * tf.matmul(s, tf.transpose(u))
    return f


class Linear(Module):
    def __init__(self,
                 batch_size=None,
                 name="linear",
                 initializer=None,
                 first=False,
                 lowest=-1,
                 highest=1,
                 alpha=2):

        self.name = name
        Module.__init__(self)

        self.batch_size = batch_size
        self.initializer = initializer
        self.first = first
        self.lowest = lowest
        self.highest = highest
        self.alpha = alpha

    def forward(self, input_tensor):
        self.input_tensor = input_tensor

        with tf.variable_scope(self.name):
            self.weights = weights_pretrained(self.initializer[0])
            self.biases = biases_pretrained(self.initializer[1])

            self.activations = tf.nn.relu(
                tf.matmul(self.input_tensor, self.weights) + self.biases
                )
        return self.activations

    def lrp(self, R):
        if self.first:
            return fprop_first_linear(
                self.input_tensor, R, self.weights, self.biases,
                self.lowest, self.highest,
                )
        else:
            return fprop_next_linear(
                self.input_tensor, R, self.weights, self.biases, self.alpha,
                )
