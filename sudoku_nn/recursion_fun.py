'''
Рекурсивный перебор.
- определить требуется ли он, если требуется, то запустить что-то типа итератора, который перебирает ячейки
- итератор должен выбрать только одну ячейку в которой 1 превратиться в 0 (SudokuIterate)
- далее идёт нерекурсивная часть
- далее четыре варианта
     1. решение существует, прекращаем работу
     2. есть прогресс в неитеративной части, надо продолжить неитеративную часть
        и запомнить прогресс используя SudokuIterateAppend.
        Потом можно будет SudokuIterateRevert использовать.
     2. решение невозможно, откатываемся на предыдущую итерацию (SudokuIterateRevert)
         (т.е. выставляем 1 в таблице возможных значений и увеличиваем recursion_index)
         Тут надо тщательно подумать как хранить предыдущие результаты. Возможно придётся какой-то
         сверхнеэффективный стек сделать на 4-5 sudoku.
     3. нет прогресса, делаем ещё один шаг рекурсии (SudokuIterate)
         (т.е. выставляем 0 в таблице возможных значений и уменьшаем recursion_index)
'''

import torch
import torch.nn as nn
from .sudoku_fun import (NNOr, NNAnd, NNNot, NNCompare, NNSelect,
                SudokuSolved, SudokuIsEqual)
import numpy as np

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

    def create_masks(self, sudoku : torch.Tensor):
        recursion_mask = torch.zeros(sudoku.shape, requires_grad=False, device=sudoku.device)
        recursion_index = torch.ones((sudoku.shape[0],), requires_grad=False, device=sudoku.device)
        return recursion_mask, recursion_index

    def forward(self, sudoku : torch.Tensor, 
                recursion_mask : torch.Tensor | None,
                recursion_index : torch.Tensor | None,
                print_removed_elem = False) -> torch.Tensor:
        
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

        if print_removed_elem:
            #print iterate
            v = one_variant.cpu().numpy()
            idx = np.unravel_index(v.argmax(), v.shape)
            print(f"SudokuIterate use n={idx[1]+1} x={idx[3]} y={idx[2]} (zero based x,y)")

        
        if True:
            #стираем всё кроме этой единички
            cell_mask = torch.clamp(self.select_number_conv(one_variant), 0, 1)
            one_recursion_mask = NNAnd(sudoku, cell_mask.expand_as(sudoku))
            one_recursion_mask = NNAnd(one_recursion_mask, NNNot(one_variant))
            add_recursion_mask = torch.mul(recursion_index.expand_as(one_recursion_mask), one_recursion_mask)
            recursion_mask = torch.fmax(recursion_mask, add_recursion_mask)

            #так-же добавляем единичку "на уровень" выше, что-бы она восстановилась при откате рекурсии
            #предполагаем, что на текущем уровне будет вызываться SudokuIterateRevert с remove_other_elems==1
            #предполагаем, что на уровне выше будет вызываться SudokuIterateRevert с remove_other_elems==0
            add_recursion_mask2 = torch.mul(torch.sub(recursion_index, 1).expand_as(one_variant), one_variant)
            recursion_mask = torch.fmax(recursion_mask, add_recursion_mask2)

            sudoku = NNAnd(sudoku, NNNot(one_recursion_mask))
            pass
        else:
            #стираем эту единичку
            add_recursion_mask = torch.mul(recursion_index.expand_as(one_variant), one_variant)
            recursion_mask = torch.add(recursion_mask, add_recursion_mask)
            sudoku = NNAnd(sudoku, NNNot(one_variant))

        return sudoku, recursion_mask, torch.add(recursion_index, 1)
    
class SudokuIterateAppend(nn.Module):
    '''
    После того как мы использовали SudokuIterate, возможно решать дальше неитеративными методами.
    И надо запомнить, что у нас нарешалось.
    Запоминать можно в той-же маске под тем-жн индексом.
    Т.е. на вход даём 2 тензора - старый и новый. Смотрим разницу.
    Очищенные места мы пишем в таблицу очищенных, а потом можно будет восстанавливать при помощи SudokuIterateRevert.
    '''
    def __init__(self, device, kernel_size=9):
        super(SudokuIterateAppend,self).__init__()

    def forward(self, sudoku_old : torch.Tensor, sudoku_new : torch.Tensor, 
                recursion_mask : torch.Tensor,
                recursion_index : torch.Tensor) -> torch.Tensor:
        clear_mask = NNCompare(torch.sub(sudoku_old, sudoku_new), 1)
        add_recursion_mask = torch.mul(clear_mask, recursion_index.expand_as(clear_mask))
        recursion_mask = torch.add(recursion_mask, add_recursion_mask)
        return recursion_mask
    
class SudokuIterateRevert(nn.Module):
    '''
    В recursion_mask ищет элемент совпадающий с recursion_index.
    Он должен быть только один для sudoku.
    Выставляем в sudoku этот элемент как 1
    Убираем эти элементы из recursion_mask
    remove_other_elems=1 - мы удаляем элементы которые находились в ячейках 
    recursion_mask где есть хотябы одно число совпадающее с recursion_index
    '''
    def __init__(self, device, kernel_size=9, remove_other_elems=False):
        super(SudokuIterateRevert,self).__init__()
        self.select_number_conv = nn.Conv2d(in_channels=kernel_size, out_channels=1, kernel_size=1)
        self.select_number_conv.weight = torch.nn.Parameter(torch.ones((1,kernel_size,1,1), device=device))
        self.select_number_conv.bias = torch.nn.Parameter(torch.zeros(1, device=device))

    def forward(self, sudoku : torch.Tensor, 
                recursion_mask : torch.Tensor,
                recursion_index : torch.Tensor,
                remove_other_elems : torch.Tensor | None) -> torch.Tensor:
        recursion_index_sub = torch.sub(recursion_index, 1)
        equal_elem = NNCompare(recursion_mask, recursion_index_sub.expand_as(recursion_mask))

        if not (remove_other_elems is None):
            clear_mask = torch.clamp(self.select_number_conv(equal_elem), 0, 1)
            clear_mask = NNAnd(clear_mask, remove_other_elems.expand_as(clear_mask))
            sudoku = NNAnd(sudoku, NNNot(clear_mask).expand_as(sudoku))

        sudoku_add = torch.fmax(sudoku, equal_elem)
        recursion_mask_sub = torch.mul(recursion_mask, NNNot(equal_elem))
        return sudoku_add, recursion_mask_sub, recursion_index_sub
    

class SudokuRecursionControl(nn.Module):
    '''
    Контролирует, нужно ли начать либо завершить рекурсию.
    - если решение судоку не продвинулось вперёд, то надо SudokuIterate вызвать (только в случае, если судоку не решено)
    - если решение судоку невалидно, то надо откатить вычеркнутое по текущей итерации.
      Если это первая итерация, то не надо откатывать текущую единицу.
    '''
    def __init__(self, device, kernel_size=9):
        super(SudokuRecursionControl,self).__init__()
        dtype = torch.float32
        self.sudoku_solved = SudokuSolved(dtype=dtype, device=device).to(device)
        self.not_equal = SudokuIsEqual(dtype=dtype, device=device).to(device)

        #!!!!!!!!!!!!SudokuIterate пока неверно работает, она убирает одну из возможностей, а должна оставлять одну из возможностей
        #делать revert для такого случая надо по особенному, убирать текущую цифру и добавлять все остальные в этой ячейке
        self.sudoku_iterate = SudokuIterate(device=device, kernel_size=kernel_size)
        self.sudoku_iterate_append = SudokuIterateAppend(device=device, kernel_size=kernel_size)
        self.sudoku_iterate_revert = SudokuIterateRevert(device=device, kernel_size=kernel_size)
        self.one = torch.tensor(1, device=device)
        
    def create_masks(self, sudoku : torch.Tensor):
        return self.sudoku_iterate.create_masks(sudoku)

    def forward(self, sudoku_old : torch.Tensor, sudoku_new : torch.Tensor, 
                recursion_mask : torch.Tensor,
                recursion_index : torch.Tensor) -> torch.Tensor:
        
        not_equal_mask = self.not_equal(sudoku_old, sudoku_new)
        is_resolved, invalid_sudoku = self.sudoku_solved(sudoku_new)
        skip_recursion = NNOr(is_resolved, not_equal_mask)
        print(f"SudokuRecursionControl is_resolved={is_resolved.item()} invalid_sudoku={invalid_sudoku.item()} skip_recursion={skip_recursion.item()}")
        if invalid_sudoku.item() > 0.5:
            pass
        if skip_recursion.item() < 0.5:
            pass

        skip_recursion = NNAnd(skip_recursion, NNNot(invalid_sudoku))

        append_recursion_mask = self.sudoku_iterate_append(sudoku_old, sudoku_new, recursion_mask, torch.sub(recursion_index, 1))
        it_sudoku, it_recursion_mask, it_recursion_index  = self.sudoku_iterate(sudoku_new, append_recursion_mask, recursion_index,
                    print_removed_elem=skip_recursion.item() < 0.5 and invalid_sudoku.item() < 0.5)
        it_recursion_index = torch.add(it_recursion_index,1) #оставляем пустой слот для sudoku_iterate_append

        is_recursion1 = NNCompare(recursion_index, 3)
        print(f"recursion_index={recursion_index.item()} is_recursion1={is_recursion1.item()}")

        reverted1_sudoku, reverted1_recursion_mask, reverted1_recursion_index = self.sudoku_iterate_revert(
                sudoku_new, append_recursion_mask, recursion_index, None)
        #если судоку невалидное, то надо таки стирать эту цифру из вариантов
        #но только в случае если первая итерация
        #Тут всё сложнее. Если судоку невалидное, то надо стирать эту цифру из вариантов, но!
        #надо её восстанавливать, когда мы откатываемся на уровень выше по итерациям
        #т.е. записывать эту цифру для индекса в котором remove_other_elems=None
        reverted2_sudoku, reverted2_recursion_mask, reverted2_recursion_index = self.sudoku_iterate_revert(
                reverted1_sudoku, reverted1_recursion_mask, reverted1_recursion_index, self.one)
        #reverted2_sudoku, reverted2_recursion_mask, reverted2_recursion_index = reverted1_sudoku, reverted1_recursion_mask, torch.sub(reverted1_recursion_index, 1)

        ret_sudoku = NNSelect(it_sudoku, reverted2_sudoku, invalid_sudoku)
        ret_recursion_mask = NNSelect(it_recursion_mask, reverted2_recursion_mask, invalid_sudoku)
        ret_recursion_index = NNSelect(it_recursion_index, reverted2_recursion_index, invalid_sudoku)

        #в случае уже решённого судоку ничего не делаем, только добавляем потёртые ячейки
        ret_sudoku = NNSelect(ret_sudoku, sudoku_new, skip_recursion)
        ret_recursion_mask = NNSelect(ret_recursion_mask, append_recursion_mask, skip_recursion)
        ret_recursion_index = NNSelect(ret_recursion_index, recursion_index, skip_recursion)
        return ret_sudoku, ret_recursion_mask, ret_recursion_index
