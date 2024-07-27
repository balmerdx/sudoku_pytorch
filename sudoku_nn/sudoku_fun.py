import torch
import torch.nn as nn
import numpy as np

def NNOr(x,y):
    return torch.clamp(torch.fmax(x,y), 0, 1)

def NNAnd(x,y):
    return torch.clamp(torch.fmin(x,y), 0, 1)

def NNNot(x):
    return torch.sub(1, x)

def NNCompare(x, value):
    #зануляем числа меньше x
    a = nn.functional.relu(torch.add(x, -value+1))
    b = nn.functional.relu(torch.sub(value+1, x))
    #зануляем числа больше x
    return torch.fmin(a,b)


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
    
class SudokuSumDownsample:
    '''
    Вспомогательный класс.
    type = "h" | "v" | "box"
    Уменьшает элементы но горизонтали/вертикали/box3x3 и суммирует.
    Потом можно обратно увеличить.
    '''
    def __init__(self, type : str, channels=9):
        if type=='h':
            #horizontal
            self.select_sum_conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(1, 9), groups=channels)
            self.select_sum_conv.weight = torch.nn.Parameter(torch.tensor([[[[1]*9]]]*channels, dtype=torch.float32))
            self.select_sum_conv.bias = torch.nn.Parameter(torch.zeros(channels, dtype=torch.float32))
            self.upsample = nn.UpsamplingNearest2d(scale_factor=(1,9))
        elif type=='v':
            #vertical
            self.select_sum_conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(9, 1), groups=channels)
            #print(f"{self.select_sum_conv.weight.shape=}")
            self.select_sum_conv.weight = torch.nn.Parameter(torch.tensor([[[[1]]*9]]*channels, dtype=torch.float32))
            #print(f"{self.select_sum_conv.weight.shape=}")
            self.select_sum_conv.bias = torch.nn.Parameter(torch.zeros(channels, dtype=torch.float32))
            self.upsample = nn.UpsamplingNearest2d(scale_factor=(9,1))
        elif type=='box':
            #box
            self.use_upsample3 = True
            self.select_sum_conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=3, groups=channels)
            self.select_sum_conv.weight = torch.nn.Parameter(torch.tensor([[[[1]*3]*3]]*channels, dtype=torch.float32))
            self.select_sum_conv.bias = torch.nn.Parameter(torch.zeros(channels, dtype=torch.float32))
            self.upsample = nn.UpsamplingNearest2d(scale_factor=3)
            pass
        else:
            #unknown type
            assert 0

    def downsample(self, mask):
        #уменьшает и суммирует
        return self.select_sum_conv(mask)
    
    def upsample(self, mask_down):
        return self.upsample(mask_down)


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

        self.sum_downsample = SudokuSumDownsample(type)

    def forward(self, mask : torch.Tensor, exact_cells : torch.Tensor) -> torch.Tensor:
        exact_cell_expanded = exact_cells.expand_as(mask)
        exact_mask = torch.mul(mask, exact_cell_expanded)

        #маска элементов которые точно встречаются на этой строке
        sum_mask_elem = self.sum_downsample.downsample(exact_mask)
        sum_mask_elem = self.sum_downsample.upsample(sum_mask_elem)

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
        self.sum_downsample = SudokuSumDownsample(type)

        self.uniq_cell_conv = nn.Conv2d(in_channels=9, out_channels=1, kernel_size=1)
        self.uniq_cell_conv.weight = torch.nn.Parameter(torch.tensor([[[[-1]]]*9], dtype=torch.float32))
        self.uniq_cell_conv.bias = torch.nn.Parameter(torch.tensor([1], dtype=torch.float32))
        self.uniq_cell_conv_relu = nn.ReLU()

    def forward(self, mask : torch.Tensor) -> torch.Tensor:
        #сумма, сколько раз элементы встречаются на этой строке
        sum_mask_elem = self.sum_downsample.downsample(mask)
        #маска элементов которые встречаются на этой строке только один раз
        uniq_mask_elem = NNCompare(sum_mask_elem, 1)

        uniq_mask_elem = self.sum_downsample.upsample(uniq_mask_elem)

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

class  SudokuIsEqual(nn.Module):
    def __init__(self):
        super(SudokuIsEqual,self).__init__()
        self.sum_all = nn.Conv2d(in_channels=9, out_channels=1, kernel_size=9)
        print(f"{self.sum_all.weight.shape=}")
        print(f"{self.sum_all.bias.shape=}")
        self.sum_all.weight = torch.nn.Parameter(torch.tensor([[[[1]*9]*9]*9], dtype=torch.float32))
        self.sum_all.bias = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, mask1 : torch.Tensor, mask2 : torch.Tensor) -> torch.Tensor:
        return self.sum_all(torch.abs(torch.sub(mask1, mask2)))

class SudokuDigitsInOneLineAtBox(nn.Module):
    '''
Если число в box находится исключительно в одном из horizontal, vertical кусков box, то значит его можно вычищать из остальных box.
Например мы хотим это для горизонтального случая сделать.
Т.е. нам надо будет пройтись по всем трём строкам и определить какие числа там находятся.
Потом у нас будет условие - что число находится в одной строке, но его нет в других.
Имея такую маску мы можем её инвертировать и пройтись по другим box в этой строке.
    '''
    def __init__(self, type):
        super(SudokuDigitsInOneLineAtBox,self).__init__()
        if type=='h':
            #horizontal
            self.pool_line = nn.MaxPool2d(kernel_size=(1,3))
            self.select_sum_opposite = nn.Conv2d(in_channels=9, out_channels=9, kernel_size=(3,1), groups=9, stride=(3,1))
            self.select_sum_opposite.weight = torch.nn.Parameter(torch.tensor([[[[-1]]*3]]*9, dtype=torch.float32))
            self.select_sum_opposite.bias = torch.nn.Parameter(torch.tensor([2]*9, dtype=torch.float32))
            self.pool_opposite = nn.MaxPool2d(kernel_size=(1,3))
            self.upsample_opposite = nn.UpsamplingNearest2d(scale_factor=(3,1))
            self.upsample_negate = nn.UpsamplingNearest2d(scale_factor=(1,3))
        elif type=='v':
            #vertical
            self.pool_line = nn.MaxPool2d(kernel_size=(3,1))
            self.select_sum_opposite = nn.Conv2d(in_channels=9, out_channels=9, kernel_size=(1,3), groups=9, stride=(1,3))
            self.select_sum_opposite.weight = torch.nn.Parameter(torch.tensor([[[[-1]*3]]]*9, dtype=torch.float32))
            self.select_sum_opposite.bias = torch.nn.Parameter(torch.tensor([2]*9, dtype=torch.float32))
            self.pool_opposite = nn.MaxPool2d(kernel_size=(3,1))
            self.upsample_opposite = nn.UpsamplingNearest2d(scale_factor=(1,3))
            self.upsample_negate = nn.UpsamplingNearest2d(scale_factor=(3,1))
        else:
            assert False

    def forward(self, mask : torch.Tensor) -> torch.Tensor:
        #маска для каждой линии в box.
        #у нас есть тройки чисел. Если два нулевых, а одно единичное, то это требуемая нам маска
        #просуммируем по этой оси и сравним с 1. 0 во всех трёх значениях теоретически не может быть.
        #это значит, что судоку не решаемо.
        line_mask = self.pool_line(mask)
        #oposite_mask единичная, если у нас есть только одна строка с этим числом
        oposite_mask = nn.functional.relu(self.select_sum_opposite(line_mask))
        oposite_mask_upsampled = self.upsample_opposite(oposite_mask)

        #фильтруем, теперь в line_mask единичные только те линии, которые уникальны
        line_mask = NNAnd(line_mask, oposite_mask_upsampled)

        #negate_mask маска ячеек, в которых надо стирать соответствующие цифры
        negate_mask_max = self.pool_opposite(line_mask).expand_as(line_mask)
        negate_mask = NNAnd(negate_mask_max, NNNot(line_mask))
        negate_mask_upsampled = self.upsample_negate(negate_mask)
        return NNAnd(mask, NNNot(negate_mask_upsampled))

class SudokuDigitsDoubles(nn.Module):
    '''
Класс, который проверяет в horizontal, vertical, box. Если есть две ячейки в которых 2 одинаковых числа,
то в этом случае значит в других ячейках этих чисел нет.

- считаем количество повторений в horizontal, vertical, box. Нам нужны числа, у которых 2 повторения.
  и вот тут возникает сложность, что делать дальше?
  Перебирать все комбинации слишком долго. Хотя может и не долго 9*8/2 = 36 комбинаций.
  Это можно себе позволить пока у нас всего 9 чисел.
  Для каждой из комбинаций 1+2, 1+3 .. 8+9 смотрим, что:
   - соответствующих чисел только 2
   - что они находятся в одной и той-же ячейке
  Используя эту маску очищаем все остальные цифры в этой ячейке.
  Далее используя эту маску можно очищать цифры вне этой ячейки.

  - маска, что есть только 2 числа в h/v/box
  - ячейки где есть только 2 числа которых 2 в h/v/box
  - for i in range(0,8): for j in range(i+1, 9): i,j = 36 штук
    Сделаем nn.Conv2d(in_channels=1, out_channels=36, kernel_size=1)
    в котором получаются суммы соответсвующих каналов и сравниваем их с 2.
  - Суммируя всё по h/v/box мы можем узнать, что таких элементов ровно 2.
    делаем UpsamplingNearest2d и фильтруем получившиеся 36 каналов для каждой ячейки
  - потом эти значения можно отфильтровать из всех оставшися в h/v/box элементов
    '''
    def __init__(self, type):
        super(SudokuDigitsDoubles,self).__init__()
        self.sum_downsample36 = SudokuSumDownsample(type, channels=36)
        self.sum_downsample9 = SudokuSumDownsample(type, channels=9)
        #print(f"{self.sum_permutations.weight.shape=}")
        #print(f"{self.sum_permutations.bias.shape=}")
        #self.sum_permutations.weight.shape=torch.Size([36, 9, 1, 1])

        all_weights = []
        count = 0
        encode_permutations_np = np.zeros((36,9))
        decode_permutations_np = np.zeros((9,36))
        
        for i in range(0,8):
            for j in range(i+1, 9):
                encode_permutations_np[count,i] = 1.
                encode_permutations_np[count,j] = 1.
                decode_permutations_np[i,count] = 1.
                decode_permutations_np[j,count] = 1.
                count += 1
        assert count==36
        encode_permutations = torch.tensor(encode_permutations_np, dtype=torch.float32).unsqueeze(2).unsqueeze(3)
        print(f"{encode_permutations.shape=}")
        decode_permutations = torch.tensor(decode_permutations_np, dtype=torch.float32).unsqueeze(2).unsqueeze(3)

        self.sum_permutations = nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1)
        self.sum_permutations.weight = torch.nn.Parameter(encode_permutations)
        self.sum_permutations.bias = torch.nn.Parameter(torch.zeros(36, dtype=torch.float32))

        self.decode_permutations = nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1)
        print(f"{self.decode_permutations.weight.shape=}")
        print(f"{self.decode_permutations.bias.shape=}")
        self.decode_permutations.weight = torch.nn.Parameter(decode_permutations)
        self.decode_permutations.bias = torch.nn.Parameter(torch.zeros(9, dtype=torch.float32))

        self.sum_all_permutations = nn.Conv2d(in_channels=36, out_channels=1, kernel_size=1)
        #print(f"{self.sum_all_permutations.weight.shape=}")
        self.sum_all_permutations.weight = torch.nn.Parameter(torch.ones((1,36,1,1), dtype=torch.float32))
        #print(f"{self.sum_all_permutations.weight.shape=}")
        self.sum_all_permutations.bias = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
        

    def forward(self, mask : torch.Tensor) -> torch.Tensor:
        #сумма по 2 элемента в этой ячейке во всех комбинациях
        permutations = self.sum_permutations(mask)
        permutations = NNCompare(permutations, 2)

        sum_permutations = self.sum_all_permutations(permutations)
        sum_permutations = NNCompare(sum_permutations, 1)
        permutations = NNAnd(permutations, sum_permutations.expand_as(permutations))
        #тут в permutations только те ячейки, в которых только 2 элемента
        #return self.decode_permutations(permutations)

        #суммируем по h/v/box
        down_permutations = self.sum_downsample36.downsample(permutations)
        #у нас сумма по 2 элемента
        down_permutations = NNCompare(down_permutations, 2)
        up_permutations = self.sum_downsample36.upsample(down_permutations)

        #тут в up_permutations уже выбранные по h/v/box наши двойки чисел
        up_permutations = NNAnd(up_permutations, NNNot(permutations))
        
        erase_mask = self.decode_permutations(up_permutations)
        #return erase_mask
        return NNAnd(mask, NNNot(erase_mask))

'''
В отдельном тестовом проекте попробовать сделать упаковку массива с кучей нулевых элементов.
Задача примерно такая у нас есть массив признаков пусто/полно + дополнительные данные.
Полные признаки надо скинуть в начало массива.
'''

'''
Класс, который проверяет (для horizontal, vertical). Если есть 3 ячейки в которых 3 одинаковых числа (и кроме них нет ничего),
то в этом случае значит в других ячейках этих чисел нет.
'''

'''
Класс, который проверяет, что sudoku решилось, либо нет возможности решить.
'''
