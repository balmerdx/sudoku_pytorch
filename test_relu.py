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

'''
device = torch.device("cuda", 0)

input = torch.round(torch.rand(1, 3, 4, device=device)*3)
print(input.cpu().numpy())

kernel_size = input.shape[2]
max_pool = nn.MaxPool1d(kernel_size, return_indices=True)
max_unpool = nn.MaxUnpool1d(kernel_size)

out = torch.tensor([[[7.77],[3.9],[4.5]]], dtype=torch.float32, device=device)
out_indices = torch.tensor([[[1],[2],[3]]], dtype=torch.int64, device=device)
print(out.cpu().numpy())
print("indices =",out_indices)
print("indices dtype =", out_indices.dtype)

out2 = max_unpool(out, out_indices)

print(out2.cpu().numpy())
'''

#sudoku = torch.round(torch.rand(1, 3, 3, 3))
sudoku = torch.ones(1, 3, 3, 3)
iterate = SudokuIterate(device=sudoku.device, kernel_size=sudoku.shape[1])
iterate_rev = SudokuIterateRevert(device=sudoku.device, kernel_size=sudoku.shape[1])
iterate_append = SudokuIterateAppend(device=sudoku.device, kernel_size=sudoku.shape[1])
print(sudoku)

sudokus = []

recursion_mask, recursion_index = None, None
iterations = 5
for i in range(iterations):
    sudokus.append(sudoku)
    sudoku, recursion_mask, recursion_index = iterate(sudoku, recursion_mask, recursion_index)
    #print(sudoku)
    sudoku_cleared_bits = torch.round(torch.mul(torch.rand(sudoku.shape),0.6))
    sudoku_cleared_bits = NNAnd(sudoku_cleared_bits, sudoku)
    #print(sudoku_cleared_bits)
    sudoku_new = NNAnd(sudoku, NNNot(sudoku_cleared_bits))
    recursion_mask = iterate_append(sudoku, sudoku_new, recursion_mask, torch.sub(recursion_index,1))
    #print(recursion_mask)
    sudoku = sudoku_new
    print(sudoku)
    pass

#print(recursion_mask)
for i in range(iterations):
    sudoku, recursion_mask, recursion_index = iterate_rev(sudoku, recursion_mask, recursion_index)
    comared = torch.max(torch.abs(torch.sub(sudokus[iterations-1-i], sudoku))).item()
    print(f"{comared=}")
    #print(recursion_mask)
    #print(sudokus[iterations-1-i])
    print(sudoku)
    pass
