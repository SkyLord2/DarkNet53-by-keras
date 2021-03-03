# -*- coding: utf-8 -*-
# @Time : 2021/2/27 16:41
# @Author : cds
# @Site : https://github.com/SkyLord2?tab=repositories
# @Email: chengdongsheng@outlook.com
# @File : DarkNet53.py
# @Software: PyCharm

import tensorflow.keras as keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, GlobalAveragePooling2D, Softmax, Dense, add, Activation

def Convolutional(input, filters, kernel_size, strides, name, padding = 'same', alpha = 0.1):
    """
    一个卷积结构由 一个卷积层，一个批归一化层，一个激活层
    :param input: 输入张量
    :param filters: 卷积核数量
    :param size: 卷积核大小
    :param stride: 步长
    :param name: 名称标识
    :return:
    """
    output = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, name=f'{name}_conv')(input)
    output = BatchNormalization(name=f'{name}_bn')(output)
    output = LeakyReLU(alpha=alpha, name=f'{name}_lr')(output)
    return output

def Residual(input, filters, name):
    output = Convolutional(input=input, filters=filters, kernel_size=(1,1), strides=1, name= f'{name}_1_')
    output = Convolutional(input=output, filters= 2 * filters, kernel_size=(3,3), strides=1, name= f'{name}_2_')
    output = add([input, output])
    output = Activation('linear')(output)
    return output

def StackResidual(input, filters, name, num):
    output = Residual(input=input, filters=filters, name=f'{name}_0_')

    for i in range(1, num):
        output = Residual(input=output, filters=filters, name=f'{name}_{i}_')
    return output


def DarkNet_Body(input):
    output = Convolutional(input=input, filters=32, kernel_size=(3,3), strides=1, name='C1')
    output = Convolutional(input=output, filters=64, kernel_size=(3,3), strides=2, name='C2')
    output = Residual(input=output, filters=32, name='R1')
    output = Convolutional(input=output, filters=128, kernel_size=(3,3), strides=3, name='C3')
    output = StackResidual(input=output, filters=64, name='S1', num=2)
    output = Convolutional(input=output, filters=256, kernel_size=(3,3), strides=2, name='C4')
    output = StackResidual(input=output, filters=128, name='S2', num=8)
    output = Convolutional(input=output, filters=512, kernel_size=(3,3), strides=2, name='C5')
    output = StackResidual(input=output, filters=256, name='S3', num=8)
    output = Convolutional(input=output, filters=1024, kernel_size=(3,3), strides=2, name='C6')
    output = StackResidual(input=output, filters=512, name='S4', num=4)
    return output


def DarkNet53():
    keras.backend.clear_session()

    input = Input(shape=(512, 512, 3))
    output = DarkNet_Body(input=input)
    output = GlobalAveragePooling2D()(output)
    output = Dense(units=1000)(output)
    output = Softmax()(output)
    model = Model(input, output)
    return model




