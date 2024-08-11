import torch
import torch.nn as nn
import numpy as np
from sudoku_stuff import *
from sudoku_nn import *

dtype=torch.float32
device = torch.device("cuda", 0)
#dtype=torch.int8


def draw(name="Initial", back_mask=None):
    print(name)
    if mask.shape[0]==1 and not(ds.prev_hints is None):
        ds.draw_sudoku(sudoku=sudoku.decode(), hints=mask_to_np(mask), store_prev_hints=False, prev_intensity=128, back_mask=back_mask)
        ds.show(time_msec=300)
        ds.draw_sudoku(sudoku=sudoku.decode(), hints=mask_to_np(mask), store_prev_hints=False, prev_intensity=192, back_mask=back_mask)
        ds.show(time_msec=300)
    ds.draw_sudoku(sudoku=sudoku.decode(), hints=mask_to_np(mask), store_prev_hints=True, use_prev_hints=False, back_mask=back_mask)
    ds.show()

#sudoku = b'9...84.6.6.4..52.7.3..7..8.76...15...53.....1...4.96.31.5.26.9...2.4....8....371.' #easy
#sudoku = b".68..5.9.7...12..6...86...287....3...92...51...3....671...83...6..59...3.5.7..18." #medium
#sudoku = b"........9..8.97.56..785.4..3..1........5.8........3..5..5.692..94.78.6..7........" #hard+
#sudoku = b"...6.2...6...4..28....1..73...7.5.81.2.....4.97.4.1...41..3....83..7...5...1.8..." #hard
#sudoku = b".....8.568...92...1.....28.7.6....1..8.9.6.4..1....9.2.48.....7...24...595.6....." #hard
#sudoku = b".......8...21.....947.3...57.62..1......5....3.17..8..183.6...4..49............6." #hard двойка одинаковых чисел
#sudoku = b'6154...8.8321.54.69476382157.628.1..4.83516..3.17.68..18356...4.649......798...6.' #hard двойка одинаковых чисел почти решённая
#sudoku = b"5..6....3.....3.2.9..2.7.8...8....32..43..9.5..9....688..7.2.5......4.7.7..9....6" #veryhard
#sudoku = b"8.........95.......76.........426798...571243...893165......916....3.487....1.532" #puzzles7_serg_benchmark extra hard
#sudoku = b".................1.....2.3......3.2...1.4......5....6..3......4.7..8...962...7..." #data/puzzles2_17_clue 0
#sudoku = b'.................1.....2.34.....4.....5...6....6.3.....3..6.....7..5.8..24......7' #data/puzzles2_17_clue 19 требуется двойка одинаковых чисел
#sudoku = b'........6.....6..1.675.29347.26.43953.572.6484.6.3517253826741967.45.82324....567' #data/puzzles2_17_clue 19 до состояния кода требуется двойка
#sudoku = b"..3.7..4...6..23.1.89.........1.7.8.517.....6...4.....271..9..5.95..........2...." #puzzles1_unbiased 0
sudoku = b"7..51...3..8...7.....4.......6....9.3...7.2...8...4..1.....26.........4.5.9.8...." #tdoku build/generate -p0 -c0 -g1 -d1 -n100 -e50 -s0

#sudoku = get_puzzle("data/puzzles0_kaggle")
#sudoku = get_puzzle("data/puzzles2_17_clue", 1)
#sudoku = get_puzzle("data/puzzles1_unbiased") #можно как hardest использовать не особо сложные
#sudoku = get_puzzle("data/puzzles5_forum_hardest_1905_11+")

#sudoku  = b'9...84.6.6.4..52.7.3..7..8.76...15...53.....1...4.96.31.5.26.9...2.4....8....371.'
#sudoku2 = b".68..5.9.7...12..6...86...287....3...92...51...3....671...83...6..59...3.5.7..18."

#https://www.google.com/search?newwindow=1&client=ubuntu-sn&hs=rzG&sca_esv=e3ff2cacd83f1ace&channel=fs&sxsrf=ADLYWIK_uox5MzPdr9OSkR06N62CrL8Crg:1722364908241&q=Fastest+algorithm+to+solve+Sudoku&uds=ADvngMjvPWL3d2w2-NRjpDFznBOP8WBJNS0ErgI6af4zFntGhDeoodfxks_hG3knjbM5AAeRyq7yXMXaK2WdmMWDuQ-0Q4zNwaO43WVGyBAwF4OQv5JwtX-VFSpSPQS7kcVJAMYhYIK4zJGKcD0qtb59Eb4FufsgKg&sa=X&ved=2ahUKEwj1052ktc-HAxUWPxAIHZ6RGUMQxKsJegQIGhAB&ictx=0&biw=2490&bih=1328&dpr=1#fpstate=ive&vld=cid:592ceb4d,vid:gP5M1EoQe-4,st:0
#sudoku = b'.....6.7..5..8.6933.......2..9..1.4...25.38...7.6..2..6.......5543.6..8..9.1.....' #easy
#sudoku = b'95....67......731.8..561...23.........9...1.........37...145..9.942......85....21' #very hard
#sudoku = b'004300209005009001070060043006002087190007400050083000600000105003508690042910300' #1 million
#sudoku = get_puzzle_csv(max_count=1000)[13]

ds = DrawSudoku(enable_store_images=True)
mask = sudoku_to_mask(sudoku, dtype=dtype)
mask = mask.to(device)

remove_h = SudokuFilterHVBox("h", dtype=dtype, device=device).to(device)
remove_v = SudokuFilterHVBox("v", dtype=dtype, device=device).to(device)
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

draw()

for idx in range(15):
    def one_pass(function, name):
        global mask
        new_mask = function(mask)
        is_equal = sudoku_equal(new_mask, mask)
        mask = new_mask
        #всегда рисуем, если у нас параллельно два судоку решается
        if is_equal.shape[0]>1 or is_equal.item() > 0:
            draw(f"{idx} {name}")

    '''
    #remove_h, remove_v видимо не нужны, они в более общем виде
    #в digits_in_one_line_at_box_h,digits_in_one_line_at_box_h обрабатываются
    one_pass(remove_h, 'remove_h')
    one_pass(remove_v, 'remove_v')
    '''

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

    draw(f"{idx} uniq_h uniq_v uniq_box", test_hints)

    if mask.shape[0]==1:
        print(hints_to_str(mask))
   
    one_pass(doubles_h, 'doubles_h')
    one_pass(doubles_v, 'doubles_v')
    one_pass(doubles_box, 'doubles_box')

    is_resolved, zeros_max = sudoku_solved(mask)
    print(f"is_resolved={is_resolved.item()} zeros_max={zeros_max.item()}")
    pass
    
