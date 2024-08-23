import torch
from sudoku_stuff import *
from sudoku_nn import *
import argparse

dtype = torch.float32
device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu', 0)

back_mask_colors = [
    ]
#показывваем только точки рекурсии
for i in range(255, 80, -70):
    back_mask_colors.append((255,100,100,i))
    back_mask_colors.append((100,255,100,0))

for i in range(255, 80, -70):
    back_mask_colors.append((255,255,100,i))
    back_mask_colors.append((100,255,100,0))


def store_sudoku_arrays():
    from os.path import join
    dir = "store_sudoku_arrays"
    torch.save(mask, join(dir, "mask"))
    torch.save(recursion_mask, join(dir, "recursion_mask"))
    torch.save(recursion_index, join(dir, "recursion_index"))
    pass

def restore_sudoku_arrays():
    from os.path import join
    global mask, recursion_mask, recursion_index
    dir = "store_sudoku_arrays"
    mask = torch.load(join(dir, "mask"), weights_only=True)
    recursion_mask = torch.load(join(dir, "recursion_mask"), weights_only=True)
    recursion_index = torch.load(join(dir, "recursion_index"), weights_only=True)
    pass

def draw(sudoku, mask, ds: DrawSudoku, name="Initial", back_mask=None, draw_witout_press_key=False):
    print(name)
    
    if mask.shape[0]==1 and not(ds.prev_hints is None):
        time_msec = 1 if draw_witout_press_key else 200
        ds.draw_sudoku(sudoku=sudoku.decode(), hints=mask_to_np(mask), store_prev_hints=False, prev_intensity=128, back_mask=back_mask, back_mask_colors=back_mask_colors)
        ds.show(time_msec=time_msec)
        ds.draw_sudoku(sudoku=sudoku.decode(), hints=mask_to_np(mask), store_prev_hints=False, prev_intensity=192, back_mask=back_mask, back_mask_colors=back_mask_colors)
        ds.show(time_msec=time_msec)
    ds.draw_sudoku(sudoku=sudoku.decode(), hints=mask_to_np(mask), store_prev_hints=True, use_prev_hints=False, back_mask=back_mask, back_mask_colors=back_mask_colors)
    key = ds.show(time_msec = 1 if draw_witout_press_key else 0)
    if key==ord('1'):
        store_sudoku_arrays()

def solve_interactive(sudoku, draw_witout_press_key=False, draw_all_changes=False,
                      enable_doubles=True,
                      enable_recursion=True):
    ds = DrawSudoku(enable_store_images=True)
    mask = sudoku_to_mask(sudoku, dtype=dtype)
    mask = mask.to(device)

    remove_box = SudokuFilterHVBox("box", dtype=dtype, device=device).to(device)
    uniq_h = SudokuUniqueHVBox("h", dtype=dtype, device=device).to(device)
    uniq_v = SudokuUniqueHVBox("v", dtype=dtype, device=device).to(device)
    uniq_box = SudokuUniqueHVBox("box", dtype=dtype, device=device).to(device)
    sudoku_equal = SudokuIsEqual(dtype=dtype, device=device).to(device)
    digits_in_one_line_at_box_h = SudokuDigitsInOneLineAtBox("h", dtype=dtype, device=device).to(device)
    digits_in_one_line_at_box_v = SudokuDigitsInOneLineAtBox("v", dtype=dtype, device=device).to(device)

    doubles_h = SudokuDigitsDoubles("v", dtype=dtype, device=device).to(device)
    doubles_v = SudokuDigitsDoubles("v", dtype=dtype, device=device).to(device)
    doubles_box = SudokuDigitsDoubles("box", dtype=dtype, device=device).to(device)

    sudoku_solved = SudokuSolved(dtype=dtype, device=device).to(device)
    sudoku_recursion = SudokuRecursionControl(device=device, enable_print=True).to(device)

    if enable_recursion:
        recursion_mask, recursion_index = sudoku_recursion.create_masks(mask)
    else:
        recursion_mask, recursion_index = None, None
    #restore_sudoku_arrays()
    #is_resolved, is_invalid = sudoku_solved(mask)
    #print(f"is_resolved={is_resolved.item()} is_invalid={is_invalid.item()}")

    draw(sudoku, mask, ds, draw_witout_press_key=draw_witout_press_key)

    for idx in range(1000):
        no_changes = True
        def one_pass(function, name):
            nonlocal mask, no_changes
            new_mask = function(mask)
            is_equal = sudoku_equal(new_mask, mask)
            mask = new_mask
            if is_equal.item() > 0 and draw_all_changes:
                no_changes = False
                draw(sudoku, mask, ds, f"{idx} {name}", draw_witout_press_key=draw_witout_press_key, back_mask=recursion_mask)

        mask_old = mask

        #remove_h, remove_v не нужны, они в более общем виде
        #в digits_in_one_line_at_box_h,digits_in_one_line_at_box_h обрабатываются
        one_pass(remove_box, 'remove_box')
        one_pass(digits_in_one_line_at_box_h, 'digits_in_one_line_at_box_h')
        one_pass(digits_in_one_line_at_box_v, 'digits_in_one_line_at_box_v')

        mask = uniq_h(mask)
        mask = uniq_v(mask)
        mask = uniq_box(mask)
        #смотрим, нет ли по doubles варианта стирания
        test_hints = doubles_box(mask, True)
        test_hints = NNOr(test_hints, doubles_v(mask, True))
        test_hints = NNOr(test_hints, doubles_h(mask, True))

        if mask.shape[0]==1:
            print(hints_to_str(mask))
    
        if enable_doubles:
            one_pass(doubles_h, 'doubles_h')
            one_pass(doubles_v, 'doubles_v')
            one_pass(doubles_box, 'doubles_box')

        if no_changes:
            draw(sudoku, mask, ds, f"{idx} free draw", draw_witout_press_key=draw_witout_press_key, back_mask=recursion_mask)

        if enable_recursion:
            new_mask, recursion_mask, recursion_index = sudoku_recursion(mask_old, mask, recursion_mask, recursion_index)
            is_equal = sudoku_equal(new_mask, mask)
            mask = new_mask
            if is_equal.item() > 0:
                draw(sudoku, mask, ds, f"{idx} sudoku_recursion", draw_witout_press_key=draw_witout_press_key, back_mask=recursion_mask)

        is_resolved, is_invalid = sudoku_solved(mask)
        if is_resolved.item()>0 and draw_witout_press_key:
            break
        pass
    
    if draw_witout_press_key:
        print("Save gif")
        ds.save_gif()

if __name__ == "__main__":
    help="""Solve sudoku. Keys on keyboard:
esc - exit
space - next step
g - save gif
s - save png
Examples:
python sudoku_interactive.py
python sudoku_interactive.py -f data/puzzles3_magictour_top1465 -fo 12
python sudoku_interactive.py -t ".....8.568...92...1.....28.7.6....1..8.9.6.4..1....9.2.48.....7...24...595.6....."
"""
    parser = argparse.ArgumentParser(description=help, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--draw_all', '-da', action='store_true', help='Показываем все шаги, где судоку изменилось.')
    parser.add_argument('--non_interactive', '-ni', action='store_true', help='Не нужно нажимать клавиш при просмотре. В конце записывает gif.')
    parser.add_argument('--file', '-f', help='Файл в котором содержится много судоку.')
    parser.add_argument('--file_offset', '-fo', type=int, default=0, help='''Какое судоку из файла решать. По умолчанию решается нулевое (самое первое судоку). Файлы с судоку находятся в папке data''')
    parser.add_argument('--text', '-t', help='Судоку в виде строки на 81 элемент')

    #draw_witout_press_key
    #parser.add_argument('--file')
    args = parser.parse_args()

    sudoku = b'9...84.6.6.4..52.7.3..7..8.76...15...53.....1...4.96.31.5.26.9...2.4....8....371.' #default easy sudoku

    if not(args.file is None):
        print(f"Use file: {args.file=} {args.file_offset}")
        sudoku = get_puzzle(args.file, args.file_offset) #easy 4 30
    if not(args.text is None):
        print(f"Use text argument '{args.text}'")
        sudoku = args.text.encode()
    #
    solve_interactive(sudoku,
                     draw_all_changes=args.draw_all,
                     draw_witout_press_key=args.non_interactive)
