import torch
import torch.nn as nn
from .sudoku_fun import (SudokuFilterHVBox, SudokuUniqueHVBox, SudokuDigitsInOneLineAtBox,
                SudokuDigitsDoubles, SudokuSolved, SudokuIsEqual)

from .recursion_fun import SudokuRecursionControl


class SudokuPasses(nn.Module):
    '''
    Класс, который может решать судоку за 1 проход.
    Если конечно повезёт и судоку не очень сложное.
    '''
    def __init__(self, dtype, device, passes=4, remove_doubles=True, recursion_control=False):
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

        if remove_doubles:
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

class SudokuRecursionPasses(nn.Module):
    '''
    Класс, который может решать судоку с рекурсией.
    Но делает это очень-очень медленно и ему нужно много-много шагов.
    '''
    def __init__(self, dtype, device, passes=32, remove_doubles=True):
        super(SudokuRecursionPasses,self).__init__()
        self.passes = passes
        self.remove_doubles = remove_doubles
        self.remove_box = SudokuFilterHVBox("box", dtype=dtype, device=device).to(device)
        self.uniq_h = SudokuUniqueHVBox("h", dtype=dtype, device=device).to(device)
        self.uniq_v = SudokuUniqueHVBox("v", dtype=dtype, device=device).to(device)
        self.uniq_box = SudokuUniqueHVBox("box", dtype=dtype, device=device).to(device)
        self.sudoku_equal = SudokuIsEqual(dtype=dtype, device=device).to(device)
        self.digits_in_one_line_at_box_h = SudokuDigitsInOneLineAtBox("h", dtype=dtype, device=device).to(device)
        self.digits_in_one_line_at_box_v = SudokuDigitsInOneLineAtBox("v", dtype=dtype, device=device).to(device)

        if remove_doubles:
            self.doubles_h = SudokuDigitsDoubles("v", dtype=dtype, device=device).to(device)
            self.doubles_v = SudokuDigitsDoubles("v", dtype=dtype, device=device).to(device)
            self.doubles_box = SudokuDigitsDoubles("box", dtype=dtype, device=device).to(device)

        self.sudoku_recursion = SudokuRecursionControl(device=device).to(device)

        self.sudoku_solved = SudokuSolved(dtype=dtype, device=device).to(device)

    def forward(self, mask : torch.Tensor) -> torch.Tensor:
        recursion_mask, recursion_index = self.sudoku_recursion.create_masks(mask)
        for _ in range(self.passes):
            mask_old = mask
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
            mask, recursion_mask, recursion_index = self.sudoku_recursion(mask_old, mask, recursion_mask, recursion_index)

        return mask, self.sudoku_solved(mask)
