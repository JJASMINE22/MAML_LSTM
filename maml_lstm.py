# -*- coding: UTF-8 -*-
'''
@Project ：MAML
@File    ：maml_lstm_v2.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import config as cfg
import tensorflow as tf
import matplotlib.pyplot as plt
from net.networks import CreateModel


class MAML:

    """
        MAML - Model-Agnostic Meta-Learning
        The constructor of this class builds and compiles the model by using customed lstm to implement the inputs with variable timing length
        as described in the paper, initializing the meta-optimizer and loading the model weights (if checkpoint exists)
    """
    def __init__(self,
                 target_size: int,
                 weight_decay: float,
                 learning_rate: list,
                 **kwargs):
        """
        :param target_size: dims of features
        :param learning_rate: lr_rates of sub and meta model
        """

        self.target_size = target_size
        self.weight_decay = weight_decay

        self.sub_lr = learning_rate[0]
        self.meta_lr = learning_rate[1]
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.meta_lr) # meta-optimizer

        self.loss = tf.keras.losses.MeanSquaredError()

        self.total_loss = []
        self.support_loss = tf.keras.metrics.Mean()
        self.query_loss = tf.keras.metrics.Mean()
        self.val_loss = tf.keras.metrics.Mean()

        self.model = CreateModel(tgt_size=self.target_size)
        self.initial_weights = self.model.get_weights()

    # @tf.function
    def forward(self, source, real, model):
        """
        This method executes a forward pass of the model using input x (model prediction).
        It uses the lossFunction to calculate the loss and returns both the loss and the predictions
        """
        pred = model(source)
        loss = self.loss(real, pred)
        for weight in model.get_kernels():
            loss += self.weight_decay*tf.reduce_sum(tf.square(weight))

        return loss, pred

    def train(self, targets):
        """
        This is the implementation of "Algorithm 2 MAML for Few-Shot Supervised Learning".
        It trains the model and plots the error over time after iterating through all the Tasks in p(T)
        a = 0.01 - Step size hyperparameter for inner-error calculation and gradient decent
        The numbers denote the step of Algorithm 2 we are currently in.
        """
        with tf.GradientTape() as query_tape:
            for target in targets:
                support_src, support_tgt, query_src, query_tgt = target
                with tf.GradientTape() as support_tape:
                    support_loss, support_logits = self.forward(support_src, support_tgt, self.model)  # Compute loss of Ti

                # Create temporary model to compute θ` - applying gradients
                sub_gradients = support_tape.gradient(support_loss, self.model.trainable_variables)

                sub_model = CreateModel(tgt_size=self.target_size)
                sub_model.set_weights(self.model.get_weights())

                """
                Manual gradient descent, convert trainable_variable to constant,
                aiming to keep the parameters of each sub model under its task fixed
                """
                sub_model.lstm1.kernel = self.model.lstm1.kernel - self.sub_lr * sub_gradients[0]
                sub_model.lstm1.recurrent_kernel = self.model.lstm1.recurrent_kernel - self.sub_lr * sub_gradients[1]
                sub_model.lstm1.bias = self.model.lstm1.bias - self.sub_lr * sub_gradients[2]

                sub_model.lstm2.kernel = self.model.lstm2.kernel - self.sub_lr * sub_gradients[3]
                sub_model.lstm2.recurrent_kernel = self.model.lstm2.recurrent_kernel - self.sub_lr * sub_gradients[4]
                sub_model.lstm2.bias = self.model.lstm2.bias - self.sub_lr * sub_gradients[5]

                sub_model.lstm3.kernel = self.model.lstm3.kernel - self.sub_lr * sub_gradients[6]
                sub_model.lstm3.recurrent_kernel = self.model.lstm3.recurrent_kernel - self.sub_lr * sub_gradients[7]
                sub_model.lstm3.bias = self.model.lstm3.bias - self.sub_lr * sub_gradients[8]

                sub_model.kernel = self.model.kernel - self.sub_lr * sub_gradients[9]

                # Sampling new points for fine-tuning
                # Calculating test-error / outer and applying gradients on original θ
                query_loss, query_logits = self.forward(query_src, query_tgt, sub_model)
                self.support_loss(support_loss)
                self.query_loss(query_loss)
                self.total_loss.append(query_loss)
            avg_query_loss = tf.reduce_mean(self.total_loss)
            meta_gradients = query_tape.gradient(avg_query_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(meta_gradients, self.model.trainable_variables))
            self.total_loss.clear()

    def test(self, source, target):
        """
        Performs a prediction on new datapoints and evaluates the prediction (loss)
        """
        loss, logits = self.forward(source, target, self.model)
        self.val_loss(loss)

    def generate_sample(self, sources, targets, batches):
        samples = self.model(sources).numpy()
        for k in range(cfg.target_size):
            plt.subplot(cfg.target_size, 1, k+1)
            plt.plot(samples[:, k], color='r', marker='*',
                     linewidth=0.5, label='feature_{:0>1d}_prediction'.format(k+1))
            plt.plot(targets[:, k], color='y', marker='o',
                     linewidth=0.5, label='feature_{:0>1d}_real'.format(k+1))
            plt.grid(True)
            plt.legend(loc='upper right', fontsize='xx-small')
        plt.savefig(fname=cfg.sample_path + '\\batch{:0>3d}.jpg'.format(batches), dpi=300)
        plt.close()
