import torch
import torch.nn as nn
import numpy as np
from sudoku_stuff import *
from sudoku_nn import *

'''
conv = nn.Conv2d(in_channels=9, out_channels=1, kernel_size=1)
print(f"{conv.weight.shape}")
print(f"{conv.bias.shape}")
exit()
'''
'''
#x = torch.tensor([0,1,0,1])
x = torch.tensor([-0.6,0.6,-0.6,0.6])
y = torch.tensor([0,0,1,1])
z = NNAnd(x,y)
exit()
'''

'''
#проверяем, что можно просуммировать 9 чисел по оси x и получить одно число на выходе
x = torch.randn(9,7,9)
conv = nn.Conv2d(in_channels=9, out_channels=9, kernel_size=(1, 9), groups=9)
print(f"{conv.weight.shape}")
print(f"{conv.bias.shape}")
y = conv(x)
y = y.expand_as(x)
print(f"{x.shape=} -> {y.shape=}")
exit()
'''

def mask_to_np(mask):
    return torch.squeeze(mask, 0).permute(1,2,0).detach().numpy()


sudoku = b"5..6....3.....3.2.9..2.7.8...8....32..43..9.5..9....688..7.2.5......4.7.7..9....6"

input = np.frombuffer(sudoku, dtype=np.int8)
input = np.reshape(input, newshape=(1,1,9,9))
input = torch.tensor(input, dtype=torch.float32)

conv_sudoku = ConvSudokuTextToBits()
mask = conv_sudoku(input)

conv_exact = SudokuNumbersInCell()
mask_exact = conv_exact(mask)

remove_h = SudokuFilterHorizontalVertical(horizontal=True)
remove_v = SudokuFilterHorizontalVertical(horizontal=False)
mask = remove_h(mask, mask_exact)
mask = remove_v(mask, mask_exact)

ds = DrawSudoku()
#ds.draw_sudoku(sudoku=sudoku.decode(), hints=mask)
ds.draw_sudoku(sudoku=None, hints=mask_to_np(mask))
ds.show()
