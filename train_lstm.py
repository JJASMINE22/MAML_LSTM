# -*- coding: UTF-8 -*-
'''
@Project ：MAML_LSTM
@File    ：train_lstm.py
@IDE     ：PyCharm
@Author  ：XinYi Huang
'''
import os
import config as cfg
import numpy as np
import tensorflow as tf
from maml_lstm import MAML
from _utils import DataGenerator

if __name__ == '__main__':

    gen = DataGenerator(txt_path=cfg.text_path,
                        time_seq=cfg.time_seq,
                        train_ratio=cfg.train_ratio,
                        task_num=cfg.task_num,
                        query_ratio=cfg.query_ratio,
                        support_query_size=cfg.support_query_size)

    maml = MAML(target_size=cfg.target_size,
                sequence_len=cfg.time_seq,
                weight_decay=cfg.weight_decay,
                learning_rate=[cfg.sub_lr_rate,
                               cfg.meta_lr_rate])

    if not os.path.exists(cfg.ckpt_path):
        os.makedirs(cfg.ckpt_path)

    ckpt = tf.train.Checkpoint(bridge=maml.model,
                               optimizer=maml.optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, cfg.ckpt_path, max_to_keep=5)

    # 如果检查点存在, 则恢复最新的检查点, 加载模型
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print('Latest checkpoint restored!!')

    train_func = gen.generate(training=True)
    test_func = gen.generate(training=False)

    for epoch in range(cfg.epoches):

        maml.train(train_func, gen.task_num)

        for i in range(gen.get_val_len()):
            source, target = next(test_func)
            maml.test(source, target)
            if np.logical_and(not i % cfg.per_sample_interval, i):
                maml.generate_sample(source, target, i)

        print(
            f'Epoch {epoch + 1}, '
            f'support_Loss: {maml.support_loss.result()}, '
            f'query_Loss:  {maml.query_loss.result()}, '
            f'val_Loss:  {maml.val_loss.result()}, '
        )
        ckpt_save_path = ckpt_manager.save()

        maml.support_loss.reset_states()
        maml.query_loss.reset_states()
        maml.val_loss.reset_states()
