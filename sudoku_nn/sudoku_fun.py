import torch
import torch.nn as nn
import numpy as np

@torch.no_grad()
def NNOr(x : torch.Tensor, y : torch.Tensor):
    return torch.clamp(torch.fmax(x,y), 0, 1)

@torch.no_grad()
def NNAnd(x : torch.Tensor, y : torch.Tensor):
    return torch.clamp(torch.fmin(x,y), 0, 1)

@torch.no_grad()
def NNNot(x : torch.Tensor):
    return torch.sub(1, x)

@torch.no_grad()
def NNCompare(x : torch.Tensor, value : float):
    #зануляем числа меньше x
    a = nn.functional.relu(torch.add(x, -value+1))
    #зануляем числа больше x
    b = nn.functional.relu(torch.sub(value+1, x))
    return torch.fmin(a,b)

def NNSelect(x0 : torch.Tensor, x1 : torch.Tensor, selector : torch.Tensor):
    '''
    Если selector==0, то выбирается x0
    Если selector==1, то выбирается x1
    '''
    return torch.lerp(x0, x1, selector)

class ConvSudokuTextToBits(nn.Module):
    '''
сконвертируем текстовое судоку из формата 
"5..6....3.....3.2.9..2.7.8...8....32..43..9.5..9....688..7.2.5......4.7.7..9....6"
в формат numpy size=(9,9,9), dtype=np.uint8, где [y,x,:] - это значения 0 или 1
для примера [1,2,3] - в ячейке с y=1,x=2 возможно существование числа 4 (т.к. у нас индекс 0 это число 1)
    '''

    def __init__(self, dtype):
        super(ConvSudokuTextToBits,self).__init__()
        self.sudoku_relu = nn.ReLU()
        self.sub_zero_char = torch.tensor([ord(b"0")], dtype=dtype)

        self.select_zero_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1)
        self.select_zero_conv.weight = torch.nn.Parameter(torch.tensor([[[[-1]]]], dtype=dtype))
        self.select_zero_conv.bias = torch.nn.Parameter(torch.tensor([1], dtype=dtype))
        self.zeros_relu = nn.ReLU()

        self.zeros_repeat_conv = nn.Conv2d(in_channels=1, out_channels=9, kernel_size=1)
        self.zeros_repeat_conv.weight = torch.nn.Parameter(torch.tensor([[[[1]]]]*9, dtype=dtype))
        self.zeros_repeat_conv.bias = torch.nn.Parameter(torch.tensor([0]*9, dtype=dtype))

        self.select_number1_conv = nn.Conv2d(in_channels=1, out_channels=9, kernel_size=1)
        self.select_number1_conv.weight = torch.nn.Parameter(torch.tensor([[[[1]]]]*9, dtype=dtype))
        self.select_number1_conv.bias = torch.nn.Parameter(torch.tensor([-x+1 for x in range(1,10)], dtype=dtype))
        self.select_number1_relu = nn.ReLU()

        self.select_number2_conv = nn.Conv2d(in_channels=1, out_channels=9, kernel_size=1)
        self.select_number2_conv.weight = torch.nn.Parameter(torch.tensor([[[[-1]]]]*9, dtype=dtype))
        self.select_number2_conv.bias = torch.nn.Parameter(torch.tensor([x+1 for x in range(1,10)], dtype=dtype))
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

class SudokuNumbersInCellOld(nn.Module):
    '''
для каждого из прямоугольников смотрим на числа от 0..9 и если у нас есть одно и только одно число (или точнее number_to_compare),
то выставляем 1. Впринципе всё элементарно - надо просуммировать и сравнить с number_to_compare.
    '''
    def __init__(self, number_to_compare, dtype):
        super(SudokuNumbersInCellOld,self).__init__()

        self.select_number1_conv = nn.Conv2d(in_channels=9, out_channels=1, kernel_size=1)
        self.select_number1_conv.weight = torch.nn.Parameter(torch.tensor([[[[1]]]*9], dtype=dtype))
        self.select_number1_conv.bias = torch.nn.Parameter(torch.tensor([-number_to_compare+1], dtype=dtype))
        self.select_number1_relu = nn.ReLU()

        self.select_number2_conv = nn.Conv2d(in_channels=9, out_channels=1, kernel_size=1)
        self.select_number2_conv.weight = torch.nn.Parameter(torch.tensor([[[[-1]]]*9], dtype=dtype))
        self.select_number2_conv.bias = torch.nn.Parameter(torch.tensor([number_to_compare+1], dtype=dtype))
        self.select_number2_relu = nn.ReLU()
        pass

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        select_number1 = self.select_number1_relu(self.select_number1_conv(x))
        select_number2 = self.select_number2_relu(self.select_number2_conv(x))
        return torch.mul(select_number1, select_number2)

class SudokuNumbersInCell(nn.Module):
    '''
для каждого из прямоугольников смотрим на числа от 0..9 и если у нас есть одно и только одно число (или точнее number_to_compare),
то выставляем 1. Впринципе всё элементарно - надо просуммировать и сравнить с number_to_compare.
    '''
    def __init__(self, number_to_compare, dtype, device):
        super(SudokuNumbersInCell,self).__init__()
        self.number_to_compare = number_to_compare

        self.select_number_conv = nn.Conv2d(in_channels=9, out_channels=1, kernel_size=1)
        self.select_number_conv.weight = torch.nn.Parameter(torch.ones((1,9,1,1), dtype=dtype, device=device))
        self.select_number_conv.bias = torch.nn.Parameter(torch.zeros(1, dtype=dtype, device=device))
        pass

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        select_number = self.select_number_conv(x)
        return NNCompare(select_number, self.number_to_compare)

class SudokuSumDownsample:
    '''
    Вспомогательный класс.
    type = "h" | "v" | "box"
    Уменьшает элементы но горизонтали/вертикали/box3x3 и суммирует.
    Потом можно обратно увеличить.
    '''
    def __init__(self, type : str, dtype, device, channels=9):
        if type=='h':
            #horizontal
            self.select_sum_conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(1, 9), groups=channels)
            self.select_sum_conv.weight = torch.nn.Parameter(torch.tensor([[[[1]*9]]]*channels, dtype=dtype, device=device))
            self.select_sum_conv.bias = torch.nn.Parameter(torch.zeros(channels, dtype=dtype, device=device))
            self.upsample = nn.UpsamplingNearest2d(scale_factor=(1,9))
        elif type=='v':
            #vertical
            self.select_sum_conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(9, 1), groups=channels)
            #print(f"{self.select_sum_conv.weight.shape=}")
            self.select_sum_conv.weight = torch.nn.Parameter(torch.tensor([[[[1]]*9]]*channels, dtype=dtype, device=device))
            #print(f"{self.select_sum_conv.weight.shape=}")
            self.select_sum_conv.bias = torch.nn.Parameter(torch.zeros(channels, dtype=dtype, device=device))
            self.upsample = nn.UpsamplingNearest2d(scale_factor=(9,1))
        elif type=='box':
            #box
            self.use_upsample3 = True
            self.select_sum_conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=3, groups=channels)
            self.select_sum_conv.weight = torch.nn.Parameter(torch.tensor([[[[1]*3]*3]]*channels, dtype=dtype, device=device))
            self.select_sum_conv.bias = torch.nn.Parameter(torch.zeros(channels, dtype=dtype, device=device))
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
    def __init__(self, type : str, dtype, device):
        super(SudokuFilterHVBox,self).__init__()

        self.numbers_in_cell = SudokuNumbersInCell(number_to_compare=1, dtype=dtype, device=device)
        self.sum_downsample = SudokuSumDownsample(type, dtype=dtype, device=device)

    def forward(self, mask : torch.Tensor) -> torch.Tensor:
        exact_cells = self.numbers_in_cell(mask)
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
    def __init__(self, type : str, dtype, device):
        super(SudokuUniqueHVBox,self).__init__()
        self.sum_downsample = SudokuSumDownsample(type, dtype=dtype, device=device)

        self.uniq_cell_conv = nn.Conv2d(in_channels=9, out_channels=1, kernel_size=1)
        self.uniq_cell_conv.weight = torch.nn.Parameter(torch.tensor([[[[-1]]]*9], dtype=dtype, device=device))
        self.uniq_cell_conv.bias = torch.nn.Parameter(torch.tensor([1], dtype=dtype, device=device))
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
    ''' Сравнивает, что две маски одинаковые.
        Т.е. решение не продвинулось дальше
        Возвращает 0 если маски одинаковые.
        '''
    def __init__(self, dtype, device):
        super(SudokuIsEqual,self).__init__()
        self.sum_all = nn.Conv2d(in_channels=9, out_channels=1, kernel_size=9)
        #print(f"{self.sum_all.weight.shape=}")
        #print(f"{self.sum_all.bias.shape=}")
        self.sum_all.weight = torch.nn.Parameter(torch.tensor([[[[1]*9]*9]*9], dtype=dtype, device=device))
        self.sum_all.bias = torch.nn.Parameter(torch.zeros(1, dtype=dtype, device=device))

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
    def __init__(self, type, dtype, device):
        super(SudokuDigitsInOneLineAtBox,self).__init__()
        if type=='h':
            #horizontal
            self.pool_line = nn.MaxPool2d(kernel_size=(1,3))
            self.select_sum_opposite = nn.Conv2d(in_channels=9, out_channels=9, kernel_size=(3,1), groups=9, stride=(3,1))
            self.select_sum_opposite.weight = torch.nn.Parameter(torch.tensor([[[[-1]]*3]]*9, dtype=dtype, device=device))
            self.select_sum_opposite.bias = torch.nn.Parameter(torch.tensor([2]*9, dtype=dtype, device=device))
            self.pool_opposite = nn.MaxPool2d(kernel_size=(1,3))
            self.upsample_opposite = nn.UpsamplingNearest2d(scale_factor=(3,1))
            self.upsample_negate = nn.UpsamplingNearest2d(scale_factor=(1,3))
        elif type=='v':
            #vertical
            self.pool_line = nn.MaxPool2d(kernel_size=(3,1))
            self.select_sum_opposite = nn.Conv2d(in_channels=9, out_channels=9, kernel_size=(1,3), groups=9, stride=(1,3))
            self.select_sum_opposite.weight = torch.nn.Parameter(torch.tensor([[[[-1]*3]]]*9, dtype=dtype, device=device))
            self.select_sum_opposite.bias = torch.nn.Parameter(torch.tensor([2]*9, dtype=dtype, device=device))
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
Очень дорогой вычислительно класс. Если всё остальное занимает 0.02 MFlops,
то этот класс использует более 2 MFlops на одну sudoku. Но он позволяет решать большую часть puzzles0_kaggle
    '''
    def __init__(self, type, dtype, device):
        super(SudokuDigitsDoubles,self).__init__()
        self.sum_downsample36 = SudokuSumDownsample(type, channels=36, dtype=dtype, device=device)

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
        encode_permutations = torch.tensor(encode_permutations_np, dtype=dtype, device=device).unsqueeze(2).unsqueeze(3)
        decode_permutations = torch.tensor(decode_permutations_np, dtype=dtype, device=device).unsqueeze(2).unsqueeze(3)

        self.encode_permutations = nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1)
        self.encode_permutations.weight = torch.nn.Parameter(encode_permutations)
        self.encode_permutations.bias = torch.nn.Parameter(torch.zeros(36, dtype=dtype, device=device))

        self.decode_permutations = nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1)
        self.decode_permutations.weight = torch.nn.Parameter(decode_permutations)
        self.decode_permutations.bias = torch.nn.Parameter(torch.zeros(9, dtype=dtype, device=device))

        self.sum_all_permutations = nn.Conv2d(in_channels=36, out_channels=1, kernel_size=1)
        self.sum_all_permutations.weight = torch.nn.Parameter(torch.ones((1,36,1,1), dtype=dtype, device=device))
        self.sum_all_permutations.bias = torch.nn.Parameter(torch.zeros(1, dtype=dtype, device=device))
        

    def forward(self, mask : torch.Tensor, return_erase=False) -> torch.Tensor:
        #сумма по 2 элемента в этой ячейке во всех комбинациях
        permutations = self.encode_permutations(mask)
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
        if return_erase:
            return erase_mask
        return NNAnd(mask, NNNot(erase_mask))

class SudokuSolved(nn.Module):
    ''' Класс, который проверяет, что sudoku решилось, либо нет возможности решить.
        Надо добавить проверку, что каждое число исключительно в единственном экземпляре в строчке/столбце/box.
        Потому как судоку может быть неверно решено.
    '''
    def __init__(self, dtype, device):
        super(SudokuSolved,self).__init__()
        self.ones_in_cell = SudokuNumbersInCell(number_to_compare=1, dtype=dtype, device=device)
        self.zeros_in_cell = SudokuNumbersInCell(number_to_compare=0, dtype=dtype, device=device)

        self.sum_9x9 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=9)
        self.sum_9x9.weight = torch.nn.Parameter(torch.ones((1,1,9,9), dtype=dtype, device=device))
        self.sum_9x9.bias = torch.nn.Parameter(torch.zeros(1, dtype=dtype, device=device))

        self.max_9x9 = nn.MaxPool2d(kernel_size=9)

        self.down_h = SudokuSumDownsample("h", dtype=dtype, device=device)
        self.down_v = SudokuSumDownsample("v", dtype=dtype, device=device)
        self.down_box = SudokuSumDownsample("box", dtype=dtype, device=device)

    def forward(self, mask : torch.Tensor) -> torch.Tensor:
        sudoku_area = mask.shape[2]*mask.shape[3]
        determine_mask = self.ones_in_cell(mask)
        #количество однозначных ячеек
        determine_sum = self.sum_9x9(determine_mask)
        #если все 81 ячейка разрешены, значит судоку решилось
        all_cells_resolved = NNCompare(determine_sum, sudoku_area).squeeze()

        zeros_mask = self.zeros_in_cell(mask)
        #если хоть в одной из ячеек ничего нельзя разместить,
        #то судоку не решилось
        zeros_max = self.max_9x9(zeros_mask)

        is_invalid = zeros_max.squeeze()

        mask_ones_in_cell = NNAnd(mask, determine_mask.expand_as(mask))

        #в каждом из h/v/box должно быть не более одного определённого числа.
        mask_h = self.down_h.downsample(mask_ones_in_cell)
        mask_h = torch.max(nn.functional.relu(torch.sub(mask_h,1)))

        mask_v = self.down_v.downsample(mask_ones_in_cell)
        mask_v = torch.max(nn.functional.relu(torch.sub(mask_v,1)))

        mask_box = self.down_box.downsample(mask_ones_in_cell)
        mask_box = torch.max(nn.functional.relu(torch.sub(mask_box,1)))

        is_resolved_invalid = NNOr(NNOr(mask_h, mask_v), mask_box)

        is_invalid = NNOr(is_invalid, is_resolved_invalid)
        is_resolved = NNAnd(all_cells_resolved, NNNot(is_resolved_invalid))

        #первое число - решено ли sudoku, второе число - решаемо ли судоку
        return is_resolved, is_invalid

'''
Этого должно хватить для решения простых hardest судоку.
Для начала сделать software класс, который разбивает решение на 2 решения.
Теоретически одно правильное, а другое неправильное должно быть.
Неправильное решение отбрасывать через определённое количество шагов.
'''

def sudoku_to_mask(sudoku, dtype):
    '''Конвертирует судоку в маску'''
    conv_sudoku = ConvSudokuTextToBits(dtype=dtype)

    if isinstance(sudoku, bytes):
        assert len(sudoku)==81
        input = np.frombuffer(sudoku, dtype=np.int8)
        input = np.reshape(input, newshape=(1,1,9,9))
        input = torch.tensor(input, dtype=dtype)
        return conv_sudoku(input)
    elif isinstance(sudoku, list):
        input = np.zeros(shape=(len(sudoku), 1, 9, 9))
        for idx, line in enumerate(sudoku):
            assert len(line)==81
            input[idx, :, :, :] = np.reshape(np.frombuffer(line, dtype=np.int8), newshape=(1,9,9))

        input = torch.tensor(input, dtype=dtype)
        return conv_sudoku(input)
    else:
        assert 0

class SudokuPasses(nn.Module):
    '''
    Класс, который может решать судоку за 1 проход.
    Если конечно повезёт и судоку не очень сложное.
    '''
    def __init__(self, dtype, device, passes=4, remove_doubles=True):
        super(SudokuPasses,self).__init__()
        self.passes = passes
        self.remove_doubles = remove_doubles
        self.remove_box = SudokuFilterHVBox("box", dtype=dtype, device=device).to(device)
        self.uniq_h = SudokuUniqueHVBox("h", dtype=dtype, device=device).to(device)
        self.uniq_v = SudokuUniqueHVBox("v", dtype=dtype, device=device).to(device)
        self.uniq_box = SudokuUniqueHVBox("box", dtype=dtype, device=device).to(device)
        self.sudoku_equal = SudokuIsEqual(dtype=dtype, device=device).to(device)
        self.digits_in_one_line_at_box_h = SudokuDigitsInOneLineAtBox("h", dtype=dtype, device=device).to(device)
        self.digits_in_one_line_at_box_v = SudokuDigitsInOneLineAtBox("v", dtype=dtype, device=device).to(device)

        self.doubles_h = SudokuDigitsDoubles("v", dtype=dtype, device=device).to(device)
        self.doubles_v = SudokuDigitsDoubles("v", dtype=dtype, device=device).to(device)
        self.doubles_box = SudokuDigitsDoubles("box", dtype=dtype, device=device).to(device)

        self.sudoku_solved = SudokuSolved(dtype=dtype, device=device).to(device)

    def forward(self, mask : torch.Tensor) -> torch.Tensor:
        for _ in range(self.passes):
            mask = self.remove_box(mask)
            mask = self.digits_in_one_line_at_box_h(mask)
            mask = self.digits_in_one_line_at_box_v(mask)
            mask = self.uniq_h(mask)
            mask = self.uniq_v(mask)
            mask = self.uniq_box(mask)

            #не требуется для решения puzzles0_kaggle
            if self.remove_doubles:
                mask = self.doubles_h(mask)
                mask = self.doubles_v(mask)
                mask = self.doubles_box(mask)
        return mask, self.sudoku_solved(mask)
