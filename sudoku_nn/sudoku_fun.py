import torch
import torch.nn as nn


def NNOr(x,y):
    return torch.clamp(torch.fmax(x,y), 0, 1)

class ConvSudokuTextToBits(nn.Module):
    '''
сконвертируем текстовое судоку из формата 
"5..6....3.....3.2.9..2.7.8...8....32..43..9.5..9....688..7.2.5......4.7.7..9....6"
в формат numpy size=(9,9,9), dtype=np.uint8, где [y,x,:] - это значения 0 или 1
для примера [1,2,3] - в ячейке с y=1,x=2 возможно существование числа 4 (т.к. у нас индекс 0 это число 1)
    '''

    def __init__(self):
        super(ConvSudokuTextToBits,self).__init__()
        self.sudoku_relu = nn.ReLU()
        self.sub_zero_char = torch.tensor([ord(b"0")], dtype=torch.float32)

        self.select_zero_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1)
        self.select_zero_conv.weight = torch.nn.Parameter(torch.tensor([[[[-1]]]], dtype=torch.float32))
        self.select_zero_conv.bias = torch.nn.Parameter(torch.tensor([1], dtype=torch.float32))
        self.zeros_relu = nn.ReLU()

        self.zeros_repeat_conv = nn.Conv2d(in_channels=1, out_channels=9, kernel_size=1)
        self.zeros_repeat_conv.weight = torch.nn.Parameter(torch.tensor([[[[1]]]]*9, dtype=torch.float32))
        self.zeros_repeat_conv.bias = torch.nn.Parameter(torch.tensor([0]*9, dtype=torch.float32))

        self.select_number1_conv = nn.Conv2d(in_channels=1, out_channels=9, kernel_size=1)
        self.select_number1_conv.weight = torch.nn.Parameter(torch.tensor([[[[1]]]]*9, dtype=torch.float32))
        self.select_number1_conv.bias = torch.nn.Parameter(torch.tensor([-x+1 for x in range(1,10)], dtype=torch.float32))
        self.select_number1_relu = nn.ReLU()

        self.select_number2_conv = nn.Conv2d(in_channels=1, out_channels=9, kernel_size=1)
        self.select_number2_conv.weight = torch.nn.Parameter(torch.tensor([[[[-1]]]]*9, dtype=torch.float32))
        self.select_number2_conv.bias = torch.nn.Parameter(torch.tensor([x+1 for x in range(1,10)], dtype=torch.float32))
        self.select_number2_relu = nn.ReLU()
        pass

    def forward(self, sudoku : torch.Tensor) -> torch.Tensor:
        sudoku = self.sudoku_relu(torch.sub(sudoku, self.sub_zero_char))
        input_zeros = self.zeros_relu(self.select_zero_conv(sudoku))
        input_zeros = self.zeros_repeat_conv(input_zeros)

        select_number1 = self.select_number1_relu(self.select_number1_conv(sudoku))
        select_number2 = self.select_number2_relu(self.select_number2_conv(sudoku))

        select_number = torch.mul(select_number1, select_number2)
        return NNOr(select_number, input_zeros)
