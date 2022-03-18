# -*- coding: UTF-8 -*-
'''
@Project ：MAML_LSTM
@File    ：CustomLayers.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
from tensorflow.keras.layers import (Add,
                                     Dense,
                                     Layer,
                                     Activation
                                     )
from tensorflow.keras import initializers
from tensorflow.keras import activations
import tensorflow as tf

class MyLSTM(Layer):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 kernel_initializer=None,
                 recurrent_initializer=None,
                 bias_initializer=None,
                 use_bias: bool=True,
                 activation=None,
                 recurrent_activation=None,
                 return_sequences: bool=False,
                 return_state: bool=False,
                 go_backwards: bool=False,
                 states: list=[],
                 **kwargs
                 ):
        super(MyLSTM, self).__init__(**kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.use_bias = use_bias
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.go_backwards = go_backwards
        self.states = states
        # kernel
        self.kernel = tf.Variable(initial_value=self.kernel_initializer(shape=(self.input_size,
                                                                               self.hidden_size*4)),
                                  name=self.name+'/kernel')
        # recurrent kernel
        self.recurrent_kernel = tf.Variable(initial_value=self.kernel_initializer(shape=(self.hidden_size,
                                                                                         self.hidden_size*4)),
                                            name=self.name+'/recurrent_kernel')
        # bias
        if self.use_bias:
            self.bias = tf.Variable(initial_value=self.bias_initializer(shape=(4*self.hidden_size,)),
                                    name=self.name+'/bias')

    def get_config(self):
        config = super(MyLSTM, self).get_config()
        config.update({
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'kernel_initializer': self.kernel_initializer,
            'recurrent_initializer': self.recurrent_initializer,
            'bias_initializer': self.bias_initializer,
            'use_bias': self.use_bias,
            'activation': self.activation,
            'recurrent_activation': self.recurrent_activation,
            'return_sequences': self.return_sequences,
            'return_state': self.return_state,
            'go_backwards': self.go_backwards,
            'states': self.states
        })
        return config

    @staticmethod
    def multiple_dot_compute(input, h, kernel, recurrent_kernel, bias, activation):

        f = tf.matmul(input, kernel) + tf.matmul(h, recurrent_kernel)
        if bias is not None:
            f = tf.nn.bias_add(f, bias)
        if activation is not None:
            f = activation(f)
        return f

    def init_state(self, batch_size):

        h_t = tf.zeros(shape=(batch_size, self.hidden_size))
        c_t = tf.zeros(shape=(batch_size, self.hidden_size))

        return h_t, c_t

    @tf.autograph.experimental.do_not_convert
    def call(self, inputs, **kwargs):

        if isinstance(inputs, list):
            assert tf.greater_equal(len(inputs), 2)
            input = inputs[0]
            batch_size, seq_len = tf.shape(input)[:-1]
            h_t = inputs[1]
            c_t = inputs[2] if len(inputs) > 2 else h_t
        else:
            input = inputs
            batch_size, seq_len = tf.shape(input)[:-1]
            h_t, c_t = self.init_state(batch_size)

        self.kernel_i, self.kernel_f, self.kernel_c, self.kernel_o = \
            tf.split(self.kernel, num_or_size_splits=4, axis=-1)

        self.recurrent_kernel_i, self.recurrent_kernel_f, \
        self.recurrent_kernel_c, self.recurrent_kernel_o = \
            tf.split(self.recurrent_kernel, num_or_size_splits=4, axis=-1)

        if self.use_bias:
            self.bias_i, self.bias_f, self.bias_c, self.bias_o = \
                tf.split(self.bias, num_or_size_splits=4, axis=0)

        out_put = []
        for i in range(seq_len):

            f_t = self.multiple_dot_compute(inputs[:, i, :], h_t, self.kernel_f,
                                            self.recurrent_kernel_f, self.bias_f,
                                            self.recurrent_activation)

            i_t = self.multiple_dot_compute(inputs[:, i, :], h_t, self.kernel_i,
                                            self.recurrent_kernel_i, self.bias_i,
                                            self.recurrent_activation)

            _c_t = self.multiple_dot_compute(inputs[:, i, :], h_t, self.kernel_c,
                                             self.recurrent_kernel_c, self.bias_c,
                                             self.activation)

            c_t = f_t * c_t + i_t * _c_t

            o_t = self.multiple_dot_compute(inputs[:, i, :], h_t, self.kernel_o,
                                            self.recurrent_kernel_o, self.bias_o,
                                            self.recurrent_activation)

            if self.activation is not None:
                c_t = self.activation(c_t)

            h_t = o_t * c_t

            out_put.append(tf.expand_dims(h_t, 1))

        out_put = tf.concat(out_put, axis=1)
        self.states.extend([h_t, c_t])
        if self.return_sequences:
            if self.return_state:
                out_put = [out_put] + self.states.copy()
        else:
            if self.return_state:
                out_put = [out_put[:, -1, :]] + self.states.copy()
            else:
                out_put = out_put[:, -1, :]
        self.states.clear()
        return out_put

    def compute_output_shape(self, input_shape):

        if self.return_sequences:
            if self.return_state:
                return tuple([(*input_shape[:2], self.hidden_size)] +
                             [(input_shape[0], self.hidden_size)] +
                             [(input_shape[0], self.hidden_size)])
            return (*input_shape[:2], self.hidden_size)

        if self.return_state:
            return ((input_shape[0], self.hidden_size), ) * 3

        return (input_shape[0], self.hidden_size)