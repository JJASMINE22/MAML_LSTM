# -*- coding: UTF-8 -*-
'''
@Project ：MAML_LSTM
@File    ：_utils.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''

import datetime
import numpy as np
import pandas as pd
from pyod.models.knn import KNN
from pyod.models.feature_bagging import FeatureBagging

class DataGenerator():
    def __init__(self,
                 txt_path,
                 time_seq,
                 train_ratio,
                 task_num,
                 query_ratio,
                 support_query_size):
        assert support_query_size >= 2
        self.txt_path = txt_path
        self.time_seq = time_seq
        self.radian = np.pi / 180
        self.train_ratio = train_ratio
        self.task_num = task_num
        self.query_ratio = query_ratio
        self.support_query_size = support_query_size
        self.split_train_val()

    def create_task(self):

        main_part_idx = np.arange(np.shape(self.train_src)[0])

        residue_num = self.support_query_size * self.task_num - np.shape(main_part_idx)[0]
        residue_part_idx = np.random.choice(np.shape(self.train_src)[0], residue_num, replace=False)

        tasks_idx = np.concatenate([main_part_idx, residue_part_idx])

        return tasks_idx

    def split_train_val(self):

        _, seq_source, target = self.preprocess()
        index = np.arange(np.shape(seq_source)[0])
        np.random.shuffle(index)
        self.train_src = seq_source[index[:int(self.train_ratio * index.__len__())]]
        self.train_tgt = target[index[:int(self.train_ratio * index.__len__())]]
        self.val_src = seq_source[index[int(self.train_ratio * index.__len__()):]]
        self.val_tgt = target[index[int(self.train_ratio * index.__len__()):]]

    @staticmethod
    def erase_default_value(x, t):

        row_index, col_index = np.where(np.equal(x, ''))
        total_index = list(np.arange(x.shape[0]))

        for default_index in list(set(row_index)):
            total_index.remove(default_index)
        return x[np.array(total_index)],\
               t[np.array(total_index)]

    @staticmethod
    def erase_anomal_value(x, t):

        clf = FeatureBagging(base_estimator=KNN(), max_features=x.shape[-1])
        clf.fit(x)
        position_index = 1 - clf.predict(x)

        return x[np.array(position_index).astype('bool')], \
               t[np.array(position_index).astype('bool')]

    def preprocess(self):
        '''
        原始数据中可能存在缺省值或畸异值
        找出索引, 并通过多项式拟合补充
        '''
        df = pd.read_excel(self.txt_path, keep_default_na=False)
        df = pd.DataFrame(data=df.values, columns=df.keys())

        time_stamp = df['时间']
        df = df.drop(columns=['城市', '时间', '天气', '风向', '风级(级)', '日降雨量(mm)', '平均总云量(%)'])

        values = df.values
        # erase default values
        values, time_stamp = self.erase_default_value(values, time_stamp)
        # erase anomal values
        values, time_stamp = self.erase_anomal_value(values, time_stamp)
        values = np.concatenate([np.array(time_stamp)[:, np.newaxis],
                                 np.array(values)], axis=-1)
        df = pd.DataFrame(data=values, columns=['时间'] + [*df.keys()])

        # recover the time stamp
        df['时间'] = df['时间'].apply(lambda x: datetime.datetime.strptime(x.split(' ')[0], '%Y-%m-%d'))
        df = df.set_index('时间')
        index = pd.date_range(df.index[0], df.index[-1], freq='D')
        df = df.reindex(index, fill_value=np.nan)
        df = df.astype(float)
        df = df.interpolate(method='polynomial', order=5)
        # divide the wind speed
        x_speed = np.array(list(map(lambda i:
                                    np.array(df['风速(m/s)'])[i] * np.cos(np.array(df['风向角度(度)'])[i] * self.radian),
                                    np.arange(df.__len__()))))

        y_speed = np.array(list(map(lambda i:
                                    np.array(df['风速(m/s)'])[i] * np.sin(np.array(df['风向角度(度)'])[i] * self.radian),
                                    np.arange(df.__len__()))))

        df.insert(loc=4, column='横向风速', value=x_speed)
        df.insert(loc=5, column='纵向风速', value=y_speed)
        df = df.drop(columns=['风速(m/s)'])

        # -1~1 normalize
        df = 2 * (df - df.min(axis=0)) / (df.max(axis=0) - df.min(axis=0)) - 1
        assign_source = np.array([np.array(df)[i:i + self.time_seq] for i in range(df.__len__() - self.time_seq)])
        assign_target = np.array([np.array(df)[i + self.time_seq] for i in range(df.__len__() - self.time_seq)])

        return np.array(df), assign_source, assign_target

    def get_val_len(self):

        base_batch_num = np.shape(self.val_src)[0]//(self.support_query_size//2)
        if not np.shape(self.val_src)[0] % (self.support_query_size//2):
            return base_batch_num
        else:
            return base_batch_num + 1

    def generate(self, training=True):
        if training:
            if np.shape(self.train_src)[0] < self.support_query_size * self.task_num:
                tasks_idx = self.create_task()
            else:
                tasks_idx = np.random.choice(np.shape(self.train_src)[0],
                                             self.support_query_size * self.task_num,
                                             replace=False)
            while True:
                if training:
                    np.random.shuffle(tasks_idx)
                    tasks_idx = np.reshape(tasks_idx, [-1, self.support_query_size])
                    for task_idx in tasks_idx:
                        support_sources = self.train_src[task_idx][:int(self.support_query_size * self.query_ratio)]
                        support_targets = self.train_tgt[task_idx][:int(self.support_query_size * self.query_ratio)]
                        query_sources = self.train_src[task_idx][int(self.support_query_size * self.query_ratio):]
                        query_targets = self.train_tgt[task_idx][int(self.support_query_size * self.query_ratio):]
                        yield support_sources, support_targets, query_sources, query_targets
        else:
            while True:
                idx = np.arange(np.shape(self.val_src)[0])
                np.random.shuffle(idx)
                val_src, val_tgt = self.val_src[idx], self.val_tgt[idx]
                sources, targets = list(), list()
                for i, (src, tgt) in enumerate(zip(val_src, val_tgt)):
                    sources.append(src)
                    targets.append(tgt)

                    if np.equal(sources.__len__(), self.support_query_size//2) or i == np.shape(val_src)[0] - 1:
                        annotation_sources, annotation_targets = sources.copy(), targets.copy()
                        sources.clear()
                        targets.clear()

                        yield np.array(annotation_sources), np.array(annotation_targets)
