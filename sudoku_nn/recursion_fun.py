'''
Для рекурсивного перебора sudoku требуется многое.

Первое - нужно определить ячейку в которую будем перебирать.
Тут нам поможет связка:
max_pool = nn.MaxPool2d(9, return_indices=True)
max_unpool = nn.MaxUnpool2d(9)
перебранные ячейки можно будет отмечать в отдельном тензоре, что-бы второй раз не перебирались.
На выходе должен быть инкрементированный индекс иекущей итерации.


Рекурсивный перебор.
- определить требуется ли он, если требуется, то запустить что-то типа итератора, который перебирает ячейки
- итератор должен выбрать только одну ячейку в которой 1 превратиться в 0
- далее идёт нерекурсивная часть
- далее три варианта
     1. решение существует, прекращаем работу
     2. решение невозможно, откатываемся на предыдущую итерацию
         (т.е. выставляем 1 в таблице возможных значений и увеличиваем recursion_index)
     3. нет прогресса, делаем ещё один шаг рекурсии
         (т.е. выставляем 0 в таблице возможных значений и уменьшаем recursion_index)
'''

import torch
import torch.nn as nn
from .sudoku_fun import NNOr, NNAnd, NNNot, NNCompare

class Iterate2D(nn.Module):
    def __init__(self, kernel_size):
        '''
        kernel_size должен совпадать с последними двумя размерностями input.shape
        '''
        super(Iterate2D,self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size, return_indices=True)
        self.max_unpool = nn.MaxUnpool2d(kernel_size)

    def forward(self, input : torch.Tensor, mask : torch.Tensor | None,
                index : torch.Tensor | None) -> torch.Tensor:
        '''
        Итерируется последовательно по числам из двухмерноно тензора input.
        input - тензор с числами. Выбираем из них максимальное и возвращаем.
                предполагаем, что все числа больше 0.
        index - текущий индекс итерации, начинается с 1. Тензор состоящий из одного float
        mask - маска чисел, которые уже были использованы.
               на место числа пишется индекс итерации.
               Изначально маска пустая.
        возвращает как data - максимальное число найденное в таблице.
        Если числа кончились, то в data возвращается 0
        '''

        if mask is None:
            mask = torch.zeros(input.shape, requires_grad=False, device=input.device)

        #Там где нули будут единицы, числа больше нуля превратятся в ноль.
        mask_neg = nn.functional.relu(torch.sub(1, mask))
        input_masked = torch.mul(input, mask_neg)
        
        data, data_indices = self.max_pool(input_masked)
        if index is None:
            index = torch.ones(data.shape, requires_grad=False, device=input.device)
        add_mask = self.max_unpool(index.expand_as(data), data_indices)
        new_mask = torch.fmax(mask, add_mask)

        return new_mask, torch.add(index, 1), data
    
class SudokuSelectOneVariant(nn.Module):
    '''
    У нас есть несколько возможных чисел в одной ячейке,
    выбираем только одно из таких чисел.
    '''
    def __init__(self, device, kernel_size=9):
        super(SudokuSelectOneVariant,self).__init__()
        self.max_pool = nn.MaxPool1d(kernel_size=kernel_size, return_indices=True)
        self.max_unpool = nn.MaxUnpool1d(kernel_size=kernel_size)

    def forward(self, sudoku : torch.Tensor, 
                current_cell_mask : torch.Tensor | None = None) -> torch.Tensor:
        sudoku_transposed = sudoku.transpose(1, 3)
        sudoku_transposed = sudoku_transposed.reshape((sudoku.shape[0], sudoku.shape[2]*sudoku.shape[3], sudoku.shape[1]))
        sudoku_filtered, sudoku_filtered_indices = self.max_pool(sudoku_transposed)
        sudoku_filtered = self.max_unpool(sudoku_filtered, sudoku_filtered_indices)

        sudoku_filtered = sudoku_filtered.reshape((sudoku.shape[0], sudoku.shape[2], sudoku.shape[3], sudoku.shape[1]))
        sudoku_filtered = sudoku_filtered.transpose(1, 3)
        
        if current_cell_mask is None:
            return sudoku_filtered
        return torch.mul(sudoku_filtered, current_cell_mask)

class SudokuIterate(nn.Module):
    '''
        Убирает с судоку 1 вариант числа в ячейке.
        Убирает только из ячеек, где есть минимум 2 числа.
        Старается выбрать ячейки с минимумом (но не менее 2).
        Если ячеек, где есть 2 числа нет, то поведение не определено.
        Заполняет recursion_mask так, что-бы потом можно было это число добавить обратно.
        recursion_index - индекс (положительное число), которое записывается в recursion_mask[j,i,k,m].
        Где j,i,k,m - индекс ячейки sudoku[j,i,k,m]=0 (в которую записали 0)

        kernel_size - можно поставить поменьше для тестов системы
    '''

    def __init__(self, device, kernel_size=9):
        super(SudokuIterate,self).__init__()
        
        self.kernel_size = kernel_size
        self.select_number_conv = nn.Conv2d(in_channels=kernel_size, out_channels=1, kernel_size=1)
        self.select_number_conv.weight = torch.nn.Parameter(torch.ones((1,kernel_size,1,1), device=device))
        self.select_number_conv.bias = torch.nn.Parameter(torch.zeros(1, device=device))

        self.max_pool = nn.MaxPool2d(kernel_size=kernel_size, return_indices=True)
        self.max_unpool = nn.MaxUnpool2d(kernel_size=kernel_size)
        self.one = torch.ones((1,1,1,1), device=device)

        self.select_one_variant = SudokuSelectOneVariant(device=device, kernel_size=kernel_size)
        pass

    def forward(self, sudoku : torch.Tensor, 
                recursion_mask : torch.Tensor | None,
                recursion_index : torch.Tensor | None) -> torch.Tensor:
        
        if recursion_mask is None:
            recursion_mask = torch.zeros(sudoku.shape, requires_grad=False, device=sudoku.device)
        if recursion_index is None:
            recursion_index = torch.ones((sudoku.shape[0],), requires_grad=False, device=sudoku.device)

        #sudoku.shape==(9,9,9)
        #numbers_in_cell.shape==(1,9,9)
        numbers_in_cell = self.select_number_conv(sudoku)
        #зануляем все, что меньше 2, 2->1, 3->2
        numbers_in_cell = nn.functional.relu(torch.sub(numbers_in_cell, 1))
        #числа равные 0 делаем равными -10, (остальные равны 0)
        numbers_in_cell_mask0 = torch.mul(NNCompare(numbers_in_cell, 0), -self.kernel_size)
        #теперь у нас -1..-7 - это числа от 2 до 9. -10 - это 0 или 1
        numbers_in_cell = torch.sub(numbers_in_cell_mask0, numbers_in_cell)

        #у нас в selected_indices - индекс ячейки, в которой требуется очистка
        selected_data, selected_indices = self.max_pool(numbers_in_cell)
        current_cell_mask = self.max_unpool(self.one.expand_as(selected_data), selected_indices)

        current_cell_mask = current_cell_mask.expand_as(sudoku)
        
        #для выбранной ячейки надо как-то выбрать одну и только одну единичку
        one_variant = self.select_one_variant(sudoku, current_cell_mask)

        add_recursion_mask = torch.mul(recursion_index.expand_as(one_variant), one_variant)
        recursion_mask = torch.add(recursion_mask, add_recursion_mask)

        sudoku = NNAnd(sudoku, NNNot(one_variant))

        return sudoku, recursion_mask, torch.add(recursion_index, 1)