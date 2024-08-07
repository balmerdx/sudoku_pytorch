from sudoku_stuff import *
from sudoku_nn import *
from time import time
from torchsummary import summary

#попробовать квантизацию (torch-tensorrt не смогло установиться :( )
dtype=torch.float32
device = torch.device("cuda", 0)
#device = torch.device("cpu", 0)
batch_size = 100000

start_time = time()
#sudoku = get_puzzle("data/puzzles0_kaggle", idx='all')
sudoku = get_puzzle_csv(max_count=batch_size)
#sudoku = get_puzzle("data/puzzles2_17_clue", idx='all')
load_time = time()-start_time
print(len(sudoku), f"load time = {load_time} sec")

start_time = time()

masks = []
for offset in range(0, len(sudoku), batch_size):
    offset_max = min(offset+batch_size, len(sudoku))
    masks.append(sudoku_to_mask(sudoku[offset:offset_max], dtype=dtype).to(device=device))
sudoku_to_mask_time = time()-start_time
print(f"to mask = {sudoku_to_mask_time} sec")

passes = SudokuPasses(dtype, device, passes=4, remove_doubles=False) #puzzles0_kaggle
#passes = SudokuPasses(dtype, device, passes=16, remove_doubles=False)
#passes = SudokuPasses(dtype, device, passes=16, remove_doubles=True) #puzzles2_17_clue
passes.eval()

with torch.no_grad():
    #summary(passes, masks[0]); exit()

    start_time = time()
    mask_solved = passes(masks[0])
    print(f"first sovled = {time()-start_time} sec")

    sum_not_solved = 0
    mask_offset = 0
    for mask in masks:
        start_time = time()
        mask_solved, solved = passes(mask)
        solve_time = time()-start_time

        is_resolved, zeros_max = solved
        is_resolved = is_resolved.detach().cpu()
        for i in range(is_resolved.shape[0]):
            if is_resolved[i] < 1.:
                print("not solved first idx=", i+mask_offset)
                break

        count_solved = int(torch.sum(is_resolved).item())
        count_failed = int(torch.sum(zeros_max.detach().cpu()).item())
        print(f"solved={count_solved}, failed={count_failed}, {solve_time:0.3f} sec, {int(mask.shape[0]/solve_time)} sudoku/sec")
        sum_not_solved += mask.shape[0]-count_solved

        mask_offset += mask.shape[0]
        pass
    percent_not_solved = sum_not_solved/len(sudoku)*100.
    percent_solved = 100-percent_not_solved
    print(f"{sum_not_solved=} {percent_solved=:.1f}%")