from sudoku_stuff import *
from sudoku_nn import *
from time import time

dtype=torch.float32
device = torch.device("cuda", 0)
#device = torch.device("cpu", 0)
batch_size = 1000

start_time = time()
sudoku = get_puzzle("data/puzzles0_kaggle", idx='all')
load_time = time()-start_time
print(len(sudoku), f"load time = {load_time} sec")

start_time = time()

masks = []
for offset in range(0, len(sudoku), batch_size):
    offset_max = min(offset+batch_size, len(sudoku))
    masks.append(sudoku_to_mask(sudoku[offset:offset_max], dtype=dtype).to(device=device))
sudoku_to_mask_time = time()-start_time
print(f"to mask = {sudoku_to_mask_time} sec")

class SudokuPasses(nn.Module):
    def __init__(self, passes=4):
        super(SudokuPasses,self).__init__()
        self.passes = passes
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
            #mask = self.doubles_h(mask)
            #mask = self.doubles_v(mask)
            #mask = self.doubles_box(mask)
        return mask, self.sudoku_solved(mask)

passes = SudokuPasses()
passes.eval()

with torch.no_grad():
    start_time = time()
    mask_solved = passes(masks[0])
    print(f"first sovled = {time()-start_time} sec")

    sum_not_solved = 0
    for mask in masks:
        start_time = time()
        mask_solved, solved = passes(mask)
        solve_time = time()-start_time

        is_resolved, zeros_max = solved
        count_solved = int(torch.sum(is_resolved.detach().cpu()).item())
        print(f"{count_solved=}, {solve_time:0.3f} sec, {int(mask.shape[0]/solve_time)} sudoku/sec")
        sum_not_solved += mask.shape[0]-count_solved
        pass
    print(f"{sum_not_solved=}")