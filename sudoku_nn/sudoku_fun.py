import torch
import torch.nn as nn

def NNOr(x,y):
    return torch.clamp(torch.fmax(x,y), 0, 1)

def NNAnd(x,y):
    return torch.clamp(torch.fmin(x,y), 0, 1)

def NNNot(x):
    return torch.sub(1, x)

def NNCompare(x, value):
    #зануляем числа меньше x
    x = nn.functional.relu(torch.sub(x, -value+1))
    #зануляем числа больше x
    return nn.functional.relu(torch.sub(value+1, x))


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
        sudoku_sub = self.sudoku_relu(torch.sub(sudoku, self.sub_zero_char))
        input_zeros = self.zeros_relu(self.select_zero_conv(sudoku_sub))
        input_zeros = self.zeros_repeat_conv(input_zeros)

        select_number1 = self.select_number1_relu(self.select_number1_conv(sudoku_sub))
        select_number2 = self.select_number2_relu(self.select_number2_conv(sudoku_sub))

        select_number = torch.mul(select_number1, select_number2)
        return NNOr(select_number, input_zeros)

class SudokuNumbersInCell(nn.Module):
    '''
для каждого из прямоугольников смотрим на числа от 0..9 и если у нас есть одно и только одно число (или точнее number_to_compare),
то выставляем 1. Впринципе всё элементарно - надо просуммировать и сравнить с number_to_compare.
    '''
    def __init__(self, number_to_compare=1.):
        super(SudokuNumbersInCell,self).__init__()

        self.select_number1_conv = nn.Conv2d(in_channels=9, out_channels=1, kernel_size=1)
        self.select_number1_conv.weight = torch.nn.Parameter(torch.tensor([[[[1]]]*9], dtype=torch.float32))
        self.select_number1_conv.bias = torch.nn.Parameter(torch.tensor([-number_to_compare+1], dtype=torch.float32))
        self.select_number1_relu = nn.ReLU()

        self.select_number2_conv = nn.Conv2d(in_channels=9, out_channels=1, kernel_size=1)
        self.select_number2_conv.weight = torch.nn.Parameter(torch.tensor([[[[-1]]]*9], dtype=torch.float32))
        self.select_number2_conv.bias = torch.nn.Parameter(torch.tensor([number_to_compare+1], dtype=torch.float32))
        self.select_number2_relu = nn.ReLU()
        pass

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        select_number1 = self.select_number1_relu(self.select_number1_conv(x))
        select_number2 = self.select_number2_relu(self.select_number2_conv(x))
        return torch.mul(select_number1, select_number2)

class SudokuFilterHVBox(nn.Module):
    '''
type = "h" | "v" | "box"

Берём точно определённые числа, находящихся на этой линии.
Из них делаем маску чисел на линии. Для этого их достаточно сложить маски ConvSudokuTextToBits
предварительно умножив на маску определённости.
Ядро для сложения должно быть 9 в длинну и 1 в высоту.

Потом имея эту маску мы стираем цифры для всех неопределённых.

По аналогии с filter horizontal/vertical мы стираем невозможые числа из box.
Вместо нужно будет воспользоваться Conv2d kernel_size=3, stride=3
И UpsamplingNearest2d вместо expand_as

    '''
    def __init__(self, type : str):
        super(SudokuFilterHVBox,self).__init__()
        
        self.use_upsample3 = False
        if type=='h':
            #horizontal
            self.select_sum_conv = nn.Conv2d(in_channels=9, out_channels=9, kernel_size=(1, 9), groups=9)
            self.select_sum_conv.weight = torch.nn.Parameter(torch.tensor([[[[1]*9]]]*9, dtype=torch.float32))
            self.select_sum_conv.bias = torch.nn.Parameter(torch.zeros(9, dtype=torch.float32))
            #print(f"{self.select_sum_conv.weight.shape=}")
            #print(f"{self.select_sum_conv.bias.shape=}")
        elif type=='v':
            #vertical
            self.select_sum_conv = nn.Conv2d(in_channels=9, out_channels=9, kernel_size=(9, 1), groups=9)
            self.select_sum_conv.weight = torch.nn.Parameter(torch.tensor([[[[1]]*9]]*9, dtype=torch.float32))
            self.select_sum_conv.bias = torch.nn.Parameter(torch.zeros(9, dtype=torch.float32))
            #print(f"{self.select_sum_conv.weight.shape=}")
            #print(f"{self.select_sum_conv.bias.shape=}")
        elif type=='box':
            #box
            self.use_upsample3 = True
            self.select_sum_conv = nn.Conv2d(in_channels=9, out_channels=9, kernel_size=3, groups=9, stride=3)
            self.select_sum_conv.weight = torch.nn.Parameter(torch.tensor([[[[1]*3]*3]]*9, dtype=torch.float32))
            self.select_sum_conv.bias = torch.nn.Parameter(torch.zeros(9, dtype=torch.float32))
            self.upsample3 = nn.UpsamplingNearest2d(scale_factor=3)
            #print(f"{self.select_sum_conv.weight.shape=}")
            #print(f"{self.select_sum_conv.bias.shape=}")
            pass
        else:
            #unknown type
            assert 0
        

    def forward(self, mask : torch.Tensor, exact_cells : torch.Tensor) -> torch.Tensor:
        exact_cell_expanded = exact_cells.expand_as(mask)
        exact_mask = torch.mul(mask, exact_cell_expanded)
        #print(f"{exact_mask.shape=}")

        #маска элементов которые точно встречаются на этой строке
        sum_mask_elem = self.select_sum_conv(exact_mask)

        if self.use_upsample3:
            sum_mask_elem = self.upsample3(sum_mask_elem)
        else:
            sum_mask_elem = sum_mask_elem.expand_as(mask)

        #маска элементов которые надо оставить
        sum_mask_pos = torch.mul(NNNot(sum_mask_elem), NNNot(exact_cell_expanded))
        #на однозначно определённых местах всегда оставляем всё
        sum_mask_pos = NNOr(sum_mask_pos, exact_cell_expanded)

        return NNAnd(sum_mask_pos, mask)

class  SudokuUniqueHVBox(nn.Module):
    '''
Класс, который проверяет. Если число есть только в одной из ячеек, то значит оно может быть только в этой ячейке.
Это даст возможность medium решать.
    '''
    def __init__(self, type : str):
        super(SudokuUniqueHVBox,self).__init__()
        self.use_upsample3 = False
        if type=='h':
            #horizontal
            self.select_sum_conv = nn.Conv2d(in_channels=9, out_channels=9, kernel_size=(1, 9), groups=9)
            self.select_sum_conv.weight = torch.nn.Parameter(torch.tensor([[[[1]*9]]]*9, dtype=torch.float32))
            self.select_sum_conv.bias = torch.nn.Parameter(torch.zeros(9, dtype=torch.float32))
            #print(f"{self.select_sum_conv.weight.shape=}")
            #print(f"{self.select_sum_conv.bias.shape=}")
        elif type=='v':
            #vertical
            self.select_sum_conv = nn.Conv2d(in_channels=9, out_channels=9, kernel_size=(9, 1), groups=9)
            self.select_sum_conv.weight = torch.nn.Parameter(torch.tensor([[[[1]]*9]]*9, dtype=torch.float32))
            self.select_sum_conv.bias = torch.nn.Parameter(torch.zeros(9, dtype=torch.float32))
            #print(f"{self.select_sum_conv.weight.shape=}")
            #print(f"{self.select_sum_conv.bias.shape=}")
        elif type=='box':
            #box
            self.use_upsample3 = True
            self.select_sum_conv = nn.Conv2d(in_channels=9, out_channels=9, kernel_size=3, groups=9, stride=3)
            self.select_sum_conv.weight = torch.nn.Parameter(torch.tensor([[[[1]*3]*3]]*9, dtype=torch.float32))
            self.select_sum_conv.bias = torch.nn.Parameter(torch.zeros(9, dtype=torch.float32))
            self.upsample3 = nn.UpsamplingNearest2d(scale_factor=3)
            #print(f"{self.select_sum_conv.weight.shape=}")
            #print(f"{self.select_sum_conv.bias.shape=}")
            pass
        else:
            #unknown type
            assert 0

        self.uniq_cell_conv = nn.Conv2d(in_channels=9, out_channels=1, kernel_size=1)
        self.uniq_cell_conv.weight = torch.nn.Parameter(torch.tensor([[[[-1]]]*9], dtype=torch.float32))
        self.uniq_cell_conv.bias = torch.nn.Parameter(torch.tensor([1], dtype=torch.float32))
        self.uniq_cell_conv_relu = nn.ReLU()

    def forward(self, mask : torch.Tensor) -> torch.Tensor:
        #сумма, сколько раз элементы встречаются на этой строке
        sum_mask_elem = self.select_sum_conv(mask)
        #маска элементов которые встречаются на этой строке только один раз
        uniq_mask_elem = NNCompare(sum_mask_elem, 1)

        if self.use_upsample3:
            uniq_mask_elem = self.upsample3(uniq_mask_elem)
        else:
            uniq_mask_elem = uniq_mask_elem.expand_as(mask)

        #выделяем ячейки в которых нет таких чисел
        not_uniq_cell_mask = torch.fmin(uniq_mask_elem, mask)
        not_uniq_cell_mask = self.uniq_cell_conv_relu(self.uniq_cell_conv(not_uniq_cell_mask))

        #общая маска - для ячеек где нет таких чисел ничего не делаем,
        #для остальных делаем and с маской uniq_mask_elem
        merged_mask = NNOr(not_uniq_cell_mask.expand_as(mask), uniq_mask_elem)

        return NNAnd(mask, merged_mask)
        #маска элементов которые надо оставить
        #sum_mask_pos = torch.mul(NNNot(sum_mask_elem), NNNot(exact_cell_expanded))
        #на однозначно определённых местах всегда оставляем всё
        #sum_mask_pos = NNOr(sum_mask_pos, exact_cell_expanded)

        #return NNAnd(sum_mask_pos, mask)

'''
Класс, который проверяет, что sudoku решилось, либо нет возможности решить.
'''

'''
3 класса, которые проверяют (для horizontal, vertical, box). Если есть две ячейки в которых 2 одинаковых числа (и кроме них нет ничего),
то в этом случае значит в других ячейках этих чисел нет.
Возможно позволить решать hard?
'''