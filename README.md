## 用于时序数据预测的正则化MAML的tensorflow2实现
---

## 目录
1. [所需环境 Environment](#所需环境) 
2. [注意事项 Attention](#注意事项)
3. [文件下载 Download](#文件下载)
4. [训练步骤 How2train](#训练步骤) 

## 所需环境
1. Python3.7
2. tensorflow-gpu>=2.0  
3. Numpy==1.19.5
4. CUDA 11.0+
5. Pandas==1.2.4
6. Pyod==0.9.8
7. Matplotlib==3.2.2

## 注意事项
1. MAML结构适用于小样本模型训练，为避免过学习，模型不应设计过重
2. sub_model的参数务必通过手动更新，附着meta_model的梯度，否则meta_model无法使用综合误差反向传递
3. 添加正则化机制，防止过拟合
4. 经多次测试，仓库https://github.com/JJASMINE22/MAML 中的MAML_LSTM网络在结合Attention的情况下，效果不佳，因此去除该机制。
5. 数据路径、训练参数自定义设置，均位于config.py
6. 更新自定义LSTM模块，使其满足渴望执行，不触发cudnn异常
7. MAML的训练任务繁多，导致梯度追踪开销巨大，训练时禁用渴望执行
8. 更新数据生成器，使MAML接受更多的训练任务

## 文件下载    
链接：https://pan.baidu.com/s/13T1Qs4NZL8NS4yoxCi-Qyw 
提取码：sets 
下载解压后放置于config.py中设置的路径即可。  

## 训练步骤
运行train_lstm.py
