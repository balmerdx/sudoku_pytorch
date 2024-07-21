from sudoku_stuff import *
import numpy as np

'''
hints = np.random.randint(low=0, high=2, size=(9,9,9), dtype=np.uint8)

hints[0,3,:] = 1
hints[3,2,:] = 0

ds = DrawSudoku()
ds.draw_sudoku(hints=hints)
ds.show()
'''

s = Sudoku()
#s.solve(sudoku='9...84.6.6.4..52.7.3..7..8.76...15...53.....1...4.96.31.5.26.9...2.4....8....371.') #easy
#s.solve(sudoku="8.........95.......76.........428967...573412...691583.....4196....3.748...8..235") #hardest
#s.solve(".68..5.9.7...12..6...86...287....3...92...51...3....671...83...6..59...3.5.7..18.") #medium
#s.solve(".91.3.28.3...4...9...8.2...25.3.4.96..9...3......7.....14...57.92..8..13..5...9..") #hard
s.solve("5..6....3.....3.2.9..2.7.8...8....32..43..9.5..9....688..7.2.5......4.7.7..9....6") #veryhard
