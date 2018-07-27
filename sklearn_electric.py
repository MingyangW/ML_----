#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 11:10:55 2018

@author: Mr Wang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plt_result(y, y_pre):
    fig, ax = plt.subplots()
    ax.scatter(y, y_pre)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()

#导入数据集：发电
data = pd.read_csv('/Users/apple/Documents/GitHub/ML_----/Folds5x2_pp.csv', )
x_data = data.iloc[:, :4]
y_data = data.iloc[:, 4]

#数据切分，交叉验证
from sklearn.cross_validation import train_test_split
x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(x_data, y_data, random_state=1)

#数据标准化
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
x_data_train_std = std.fit_transform(x_data_train)
#x_data_train_std = StandardScaler.fit_transform(X=x_data_train)
x_data_test_std = std.transform(x_data_test)

#******************************************************************************
#线性回归模型
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_data_train_std, y_data_train)

y_pre = lr.predict(x_data_test_std)

from sklearn import metrics
print("MSE_LR:", metrics.mean_squared_error(y_pre, y_data_test))
#plt_result(y_data_test, y_pre)

#******************************************************************************
#神经网络实现回归算法

y_data_train_r = y_data_train.values.reshape([len(y_data_train), 1])
y_data_test_r = y_data_test.values.reshape([len(y_data_test), 1])

import tensorflow as tf

x_tf = tf.placeholder(tf.float32, [None, 4])
y_tf = tf.placeholder(tf.float32, [None, 1])

#一个隐藏层，连接结构 4->10->1,输出层不做激活函数转换
W_1 = tf.Variable(tf.truncated_normal([4, 10], stddev=0.1))
b_1 = tf.Variable(tf.constant(0.1, shape=[10]))
layer_1 = tf.nn.relu(tf.matmul(x_tf, W_1) + b_1)

W_2 = tf.Variable(tf.truncated_normal([10, 1], stddev=0.1))
b_2 = tf.Variable(tf.constant(0.1, shape=[1]))
prediction = tf.matmul(layer_1, W_2) + b_2

loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_tf - prediction), reduction_indices=[1]))

train = tf.train.AdamOptimizer(0.5).minimize(loss)

init = tf.global_variables_initializer()

#计算MSE
def compute_accuracy(x, y):
    global prediction
    pre = sess.run(prediction, feed_dict={x_tf:x})
    output = metrics.mean_squared_error(pre, y)
    return output

with tf.Session() as sess:
    sess.run(init)
    for i in range(4000):
        sess.run(train, feed_dict={x_tf:x_data_train_std, y_tf:y_data_train_r})
#        if i % 1000==0:
#            print('MSE_NN: ', compute_accuracy(x_data_test_std, y_data_test))
    print('MSE_NN: ', compute_accuracy(x_data_test_std, y_data_test_r))


#******************************************************************************
#GBRT
from sklearn.ensemble import GradientBoostingRegressor
gbrt = GradientBoostingRegressor(n_estimators=100)
gbrt.fit(x_data_train_std, y_data_train)
y_pre = gbrt.predict(x_data_test_std)
print("MSE_GBRT:", metrics.mean_squared_error(y_pre, y_data_test))
#plt_result(y_data_test, y_pre)

#******************************************************************************
#Ridge
from sklearn.linear_model import RidgeCV
ridge = RidgeCV(alphas=[0.01, 0.1, 1, 3, 5, 8, 10, 100, 1000])
ridge.fit(x_data_train_std, y_data_train)
y_pre = ridge.predict(x_data_test_std)
print("MSE_Ridge:", metrics.mean_squared_error(y_pre, y_data_test))