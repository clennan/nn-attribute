
import tensorflow as tf
from src.modules.module import Module


class Reshape(Module):
    def __init__(self, reshape_to=None, name='reshape'):
        self.name = name
        if reshape_to is not None:
            self.reshape_to = reshape_to
        else:
            self.reshape_to = -1
        Module.__init__(self)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.dims = self.input_tensor.get_shape().as_list()
        with tf.name_scope(self.name):
            self.activations = tf.reshape(
                self.input_tensor, [self.dims[0], self.reshape_to],
                )
        return self.activations

    def lrp(self, R, *args, **kwargs):
        return tf.reshape(R, self.dims)
