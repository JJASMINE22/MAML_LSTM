# -*- coding: UTF-8 -*-
'''
@Project ：CNN_LSTM
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import tensorflow as tf
import config as cfg
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Attention
from tensorflow.keras import initializers
from CustomLayers import MyLSTM


class CreateModel(Model):
    def __init__(self,
                 tgt_size=cfg.target_size,
                 **kwargs):
        super(CreateModel, self).__init__(**kwargs)
        self.lstm1 = MyLSTM(input_size=tgt_size,
                            hidden_size=64,
                            kernel_initializer='random_normal',
                            recurrent_initializer='random_normal',
                            bias_initializer='zeros',
                            activation='tanh',
                            recurrent_activation='sigmoid',
                            return_sequences=True,
                            return_state=False)

        self.lstm2 = MyLSTM(input_size=64,
                            hidden_size=128,
                            kernel_initializer='random_normal',
                            recurrent_initializer='random_normal',
                            bias_initializer='zeros',
                            activation='tanh',
                            recurrent_activation='sigmoid',
                            return_sequences=True,
                            return_state=False)

        self.lstm3 = MyLSTM(input_size=128,
                            hidden_size=256,
                            kernel_initializer='random_normal',
                            recurrent_initializer='random_normal',
                            bias_initializer='zeros',
                            activation='tanh',
                            recurrent_activation='sigmoid',
                            return_sequences=False)

        self.kernel = tf.Variable(initial_value=initializers.get('random_normal')(shape=(256, tgt_size)),
                                  name='dense/kernel', dtype=tf.float32)

    def get_kernels(self):

        weights = []
        weights.append(self.lstm1.kernel)
        weights.append(self.lstm2.kernel)
        weights.append(self.lstm3.kernel)
        weights.append(self.kernel)

        return weights

    def call(self, input, training=None, mask=None):

        x = self.lstm1(input)
        x = tf.nn.dropout(x, rate=0.3)

        x = self.lstm2(x)
        x = tf.nn.dropout(x, rate=0.3)

        x = self.lstm3(x)
        x = tf.nn.dropout(x, rate=0.3)

        output = tf.matmul(x, self.kernel)

        return output
