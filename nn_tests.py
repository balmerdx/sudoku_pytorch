import torch
import torch.nn as nn
import numpy as np
from sudoku_stuff import *
from sudoku_nn import *

sudoku = b"5..6....3.....3.2.9..2.7.8...8....32..43..9.5..9....688..7.2.5......4.7.7..9....6"

input = np.frombuffer(sudoku, dtype=np.int8)
input = np.reshape(input, newshape=(1,1,9,9))
input = torch.tensor(input, dtype=torch.float32)

conv_sudoku = ConvSudokuTextToBits()
mask = conv_sudoku(input)
mask = torch.squeeze(mask, 0).permute(1,2,0).detach().numpy() #

ds = DrawSudoku()
#ds.draw_sudoku(sudoku=sudoku.decode(), hints=mask)
ds.draw_sudoku(sudoku=None, hints=mask)
ds.show()
