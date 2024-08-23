from sudoku_stuff import *
from sudoku_nn import *
from time import time
from torchsummary import summary

def speed_test(filename="data/puzzles0_kaggle", batch_size=100000,
               passes_count=4, remove_doubles=False, use_recursion=False):
    dtype=torch.float32
    device = torch.device("cuda", 0)
    #device = torch.device("cpu", 0)

    start_time = time()
    #sudoku = get_puzzle_csv(max_count=batch_size)
    sudoku = get_puzzle(filename, idx='all')
    load_time = time()-start_time
    print(len(sudoku), f"load time = {load_time} sec, puzzles count = {len(sudoku)} ")
    #if len(sudoku) > 2*batch_size:
    #    #нам не требуется гиганское количество для тестов
    #    sudoku = sudoku[0:2*batch_size]

    start_time = time()

    masks = []
    for offset in range(0, len(sudoku), batch_size):
        offset_max = min(offset+batch_size, len(sudoku))
        masks.append(sudoku_to_mask(sudoku[offset:offset_max], dtype=dtype).to(device=device))

    sudoku_to_mask_time = time()-start_time
    print(f"to mask = {sudoku_to_mask_time} sec")

    if use_recursion:
        passes = SudokuRecursionPasses(dtype, device, passes=passes_count, remove_doubles=remove_doubles)
    else:
        passes = SudokuPasses(dtype, device, passes=passes_count, remove_doubles=remove_doubles)
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
            mask, solved = passes(mask)
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

if __name__ == "__main__":
    #speed_test("data/puzzles0_kaggle", batch_size=100000, passes_count=4, remove_doubles=False)
    #speed_test("data/puzzles2_17_clue", batch_size=100000, passes_count=16, remove_doubles=True)
    #speed_test("data/puzzles2_17_clue", batch_size=100000, passes_count=32, remove_doubles=True)
    #speed_test("data/puzzles7_serg_benchmark", batch_size=10000, passes_count=50, remove_doubles=True, use_recursion=True)
    speed_test("data/puzzles2_17_clue", batch_size=10000, passes_count=50, remove_doubles=True, use_recursion=True)
    pass