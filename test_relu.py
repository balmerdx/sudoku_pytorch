import numpy as np
import torch
import torch.nn as nn
from sudoku_nn import *

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

'''

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
'''

'''
count = 0
for i in range(0,8):
    for j in range(i+1, 9):
        print(i,j)
        count += 1
print(count)
'''

'''
input = torch.rand(1, 1, 9, 9)
print(input.numpy())

max_pool = nn.MaxPool2d(9, return_indices=True)
max_unpool = nn.MaxUnpool2d(9)
out, out_indices = max_pool(input)
print(out.numpy())
print("indices =",out_indices)
print("indices type =",type(out_indices))

out2 = max_unpool(out, out_indices)

print(out2.numpy())
'''

'''
input = torch.mul(torch.rand(1, 1, 3, 3), 10)
iterator = Iterate2D(kernel_size=input.shape[3])
mask, index = None, None
for i in range(10):
    mask, index, data = iterator(input, mask, index)
    print(f"data={data.item()} index={index.item()-1}")
    print(mask.numpy())

'''

'''
input = torch.rand(1, 3, 3, 3)
input = torch.round(input)
one_variant = SudokuSelectOneVariant(device=input.device, kernel_size=input.shape[1])
print(input)
out = one_variant(input)
print(out)
'''

sudoku = torch.round(torch.rand(1, 3, 3, 3))
iterate = SudokuIterate(device=sudoku.device, kernel_size=sudoku.shape[1])
print(sudoku)

recursion_mask, recursion_index = None, None
for i in range(16):
    sudoku, recursion_mask, recursion_index = iterate(sudoku, recursion_mask, recursion_index)
    print(sudoku)
