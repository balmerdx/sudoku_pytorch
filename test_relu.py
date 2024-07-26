'''
#Как при помощи relu(+-x+bias) для конкретного числа получить 1?

def relu(x:float): return max(x,0.)


def f(x):
    #все числа меньше expected считаем 0
    x1 = relu(x-expected+1)
    #все числа больше expected считаем 0
    x2 = relu(-x+expected+1)
    print(x, x1, x2, x1*x2)

expected = 3

for x in range(6):
    f(float(x))

'''

import numpy as np
import torch
import torch.nn as nn
from sudoku_nn import *

#negate_mask единичная, если у нас есть только одна строка (в box) с этим числом
#надо переносить маску на другие box горизонтально, т.е. для первого столбца перенести во второй и третий
#для второго - в первый и третий. для третьего - в первый и второй.

line_mask_np = np.zeros(shape=(1,9,9,3))

for i in range(3):
    line_mask_np[0,i,i,i] = 1.

line_mask = torch.Tensor(line_mask_np)

pool = nn.MaxPool2d(kernel_size=(1,3))

#print(f"{permute_conv.weight.shape=}")
#print(f"{permute_conv.bias.shape=}")

negate_mask_max = pool(line_mask).expand_as(line_mask)
#маска ячеек, в которых надо стирать соответствующие цифры
negate_mask = NNAnd(negate_mask_max, NNNot(line_mask))
print(f"{line_mask.shape=} -> {negate_mask.shape=}")
pass