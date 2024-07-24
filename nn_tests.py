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

def draw():
    ds.draw_sudoku(sudoku=None, hints=mask_to_np(mask))
    ds.show()


#sudoku = b'9...84.6.6.4..52.7.3..7..8.76...15...53.....1...4.96.31.5.26.9...2.4....8....371.' #easy
sudoku = b".68..5.9.7...12..6...86...287....3...92...51...3....671...83...6..59...3.5.7..18." #medium
#sudoku = b"........9..8.97.56..785.4..3..1........5.8........3..5..5.692..94.78.6..7........" #hard
#sudoku = b"5..6....3.....3.2.9..2.7.8...8....32..43..9.5..9....688..7.2.5......4.7.7..9....6" #veryhard

ds = DrawSudoku()
input = np.frombuffer(sudoku, dtype=np.int8)
input = np.reshape(input, newshape=(1,1,9,9))
input = torch.tensor(input, dtype=torch.float32)

conv_sudoku = ConvSudokuTextToBits()
mask = conv_sudoku(input)

conv_exact = SudokuNumbersInCell()
remove_h = SudokuFilterHVBox("h")
remove_v = SudokuFilterHVBox("v")
remove_box = SudokuFilterHVBox("box")
uniq_h = SudokuUniqueHVBox("h")
uniq_v = SudokuUniqueHVBox("v")
uniq_box = SudokuUniqueHVBox("box")

draw() 

for _ in range(5):
    mask_exact = conv_exact(mask)
    mask = remove_h(mask, mask_exact)
    draw()
    mask_exact = conv_exact(mask)
    mask = remove_v(mask, mask_exact)
    draw()
    mask_exact = conv_exact(mask)
    mask = remove_box(mask, mask_exact)
    draw()
    mask = uniq_h(mask)
    mask = uniq_v(mask)
    mask = uniq_box(mask)
    draw()
 
pass

#ds.draw_sudoku(sudoku=sudoku.decode(), hints=mask)
