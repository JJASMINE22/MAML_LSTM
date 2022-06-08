# -*- coding: UTF-8 -*-
'''
@Project ：MAML_LSTM
@File    ：CustomLayers.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers
from tensorflow.keras import activations
import tensorflow as tf

class MyLSTM(Layer):
    """
    Customed LSTM
    With the exception of the meta-learning
    Eager execution can be enabled without triggering cudnn exception
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 sequence_len: int,
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
        self.seq_len = sequence_len
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
            'seq_len': self.seq_len,
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
            assert inputs.__len__() >= 2
            feats = inputs[0]
            batch_size = tf.shape(feats)[0]
            h_t = inputs[1]
            c_t = inputs[2] if inputs.__len__() > 2 else h_t
        else:
            feats = inputs
            batch_size = tf.shape(feats)[0]
            h_t, c_t = self.init_state(batch_size)

        feats = tf.split(feats, num_or_size_splits=self.seq_len, axis=1)

        self.kernel_i, self.kernel_f, self.kernel_c, self.kernel_o = \
            tf.split(self.kernel, num_or_size_splits=4, axis=-1)

        self.recurrent_kernel_i, self.recurrent_kernel_f, \
        self.recurrent_kernel_c, self.recurrent_kernel_o = \
            tf.split(self.recurrent_kernel, num_or_size_splits=4, axis=-1)

        if self.use_bias:
            self.bias_i, self.bias_f, self.bias_c, self.bias_o = \
                tf.split(self.bias, num_or_size_splits=4, axis=0)

        out_put = list()
        for feat in feats:

            f_t = self.multiple_dot_compute(feat, h_t, self.kernel_f,
                                            self.recurrent_kernel_f, self.bias_f,
                                            self.recurrent_activation)

            i_t = self.multiple_dot_compute(feat, h_t, self.kernel_i,
                                            self.recurrent_kernel_i, self.bias_i,
                                            self.recurrent_activation)

            _c_t = self.multiple_dot_compute(feat, h_t, self.kernel_c,
                                             self.recurrent_kernel_c, self.bias_c,
                                             self.activation)

            c_t = f_t * c_t + i_t * _c_t

            o_t = self.multiple_dot_compute(feat, h_t, self.kernel_o,
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

class EagerLSTM(Layer):
    """
    Customed LSTM
    With the exception of the meta-learning
    Eager execution can be enabled without triggering cudnn exception
    """
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
        super(EagerLSTM, self).__init__(**kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        if kernel_initializer:
            self.kernel_initializer = initializers.get(kernel_initializer)
        else:
            self.kernel_initializer = initializers.glorot_uniform
        if recurrent_initializer:
            self.recurrent_initializer = initializers.get(recurrent_initializer)
        else:
            self.recurrent_initializer = initializers.glorot_uniform
        if bias_initializer:
            self.bias_initializer = initializers.get(bias_initializer)
        else:
            self.bias_initializer = initializers.zeros
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
        config = super(EagerLSTM, self).get_config()
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

    def init_state(self, batch_size):

        h_t = tf.zeros(shape=(batch_size, self.hidden_size))
        c_t = tf.zeros(shape=(batch_size, self.hidden_size))

        return h_t, c_t

    # @tf.autograph.experimental.do_not_convert
    @tf.function
    def call(self, inits, **kwargs):
        """
        Especially designed for eager execution
        Use while_loop() to create a graph that support variable-length time series inputs
        """
        assert isinstance(inits, list) or isinstance(inits, tf.Tensor)
        if isinstance(inits, list):
            assert inits.__len__() == 3 and inits[0].shape.__len__() == 3
            feats, h_t, c_t = inits
            # enable while using bidirectional lstm
            if self.go_backwards:
                feats = feats[:, ::-1]
            batch_size, seq_len = tf.shape(feats)[0], tf.shape(feats)[1]
        else:
            assert inits.shape.__len__() == 3
            feats = inits
            # enable while using bidirectional lstm
            if self.go_backwards:
                feats = feats[:, ::-1]
            batch_size, seq_len = tf.shape(feats)[0], tf.shape(feats)[1]
            h_t, c_t = self.init_state(batch_size)
        activation = self.activation
        recurrent_activation = self.recurrent_activation

        def linear_compute(input, h, kernel, recurrent_kernel, bias, is_recurrent=True):

            f = tf.matmul(input, kernel) + tf.matmul(h, recurrent_kernel)
            if bias is not None:
                f = tf.nn.bias_add(f, bias)
            if is_recurrent:
                f = recurrent_activation(f)
            else:
                f = activation(f)
            return f

        kernel_i, kernel_f, kernel_c, kernel_o = \
            tf.split(self.kernel, num_or_size_splits=4, axis=-1)

        recurrent_kernel_i, recurrent_kernel_f, \
        recurrent_kernel_c, recurrent_kernel_o = \
            tf.split(self.recurrent_kernel, num_or_size_splits=4, axis=-1)

        if self.use_bias:
            bias_i, bias_f, bias_c, bias_o = \
                tf.split(self.bias, num_or_size_splits=4, axis=0)

        out_put = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        def loop_body(i, out_put, h_t, c_t):

            f_t = linear_compute(feats[:, i], h_t, kernel_f,
                                 recurrent_kernel_f, bias_f)

            i_t = linear_compute(feats[:, i], h_t, kernel_i,
                                 recurrent_kernel_i, bias_i)

            _c_t = linear_compute(feats[:, i], h_t, kernel_c,
                                  recurrent_kernel_c, bias_c,
                                  is_recurrent=False)

            c_t = f_t * c_t + i_t * _c_t

            o_t = linear_compute(feats[:, i], h_t, kernel_o,
                                 recurrent_kernel_o, bias_o)

            c_t = activation(c_t)

            h_t = o_t * c_t

            out_put = out_put.write(i, h_t)

            return i + 1, out_put, h_t, c_t

        _, out_put, h_t, c_t = tf.while_loop(lambda i, *args: i < seq_len, loop_body,
                                             [0, out_put, h_t, c_t])
        out_put = out_put.stack()
        out_put = tf.transpose(out_put, [1, 0, 2])

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
        """
        :param input_shape: shape of input
        :return: shape of output
        """
        if self.return_sequences:
            if self.return_state:
                return tuple([(*input_shape[:2], self.hidden_size)] +
                             [(input_shape[0], self.hidden_size)] +
                             [(input_shape[0], self.hidden_size)])
            return (*input_shape[:2], self.hidden_size)

        if self.return_state:
            return ((input_shape[0], self.hidden_size), ) * 3

        return (input_shape[0], self.hidden_size)