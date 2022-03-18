# -*- coding: UTF-8 -*-
'''
@Project ：bridge
@File    ：config.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
# ===bridge===

# data_generator
text_path = '数据文件绝对路径'
task_num = 20
time_seq = 7
query_ratio=0.5
train_ratio = 0.7
support_query_size = 64

# model
target_size = 9 # depends on the setting of the data prediction dimensions
weight_decay = 5e-4
dropout = 0.3

# training
epoches = 300
meta_lr_rate = 1e-3  # meta学习率
sub_lr_rate = 1e-3  # sub学习率
per_sample_interval = 10
ckpt_path = '模型存储路径'
sample_path = '预测效果存储路径'
