import numpy as np
from .sudoku_draw import DrawSudoku

class Sudoku:
    def __init__(self):
        self._make_sequaences()
        pass
    def solve(self, sudoku, interactive=True):
        if interactive:
            ds = DrawSudoku()
        self.hints = np.ones((9,9,9), dtype=np.uint8)
        self._fill_hints(sudoku)

        for i in range(100):
            print(i)
            if not self._filter_hints():
                if not self._filter_hints2_0():
                    if not self._filter_hints2():
                        self._filter_hints3()
                        pass
            if interactive:
                ds.draw_sudoku(sudoku, self.hints)
                ds.show()
            if self.is_solved():
                break
        pass

    def _fill_hints(self, sudoku : str):
        assert(len(sudoku)==81)
        for i in range(81):
            s = sudoku[i]
            x = i%9
            y = i//9
            if s.isdigit():
                self.hints[y,x,:] = 0
                self.hints[y,x,int(s)-1] = 1

    def is_only_one(self, x,y):
        '''
        return Bool, Int
        True - Однозначно оределенно
        False - неоднозначно
        Int - индекс нашего числа (начинается с нуля)
        '''
        cell_hints = self.hints[y,x,:]
        hint_sum = 0
        for hi in range(9):
            hint_sum += 1 if cell_hints[hi] else 0
        if hint_sum == 0:
            return True, -1 #Решение не существует!
        if hint_sum != 1:
            return False, 0
        for hi in range(9):
            if cell_hints[hi]:
                return True, hi
        pass

    def is_double(self, x,y):
        ''' Если это двойка чисел, то возвращаем её.
        Иначе возвращаем None
        '''
        cell_hints = self.hints[y,x,:]
        hint_sum = 0
        for hi in range(9):
            hint_sum += 1 if cell_hints[hi] else 0
        if hint_sum!=2:
            return None
        arr = []
        for hi in range(9):
            if cell_hints[hi]:
                arr.append(hi)
        assert len(arr)==2
        return tuple(arr)

    def is_solved(self):
        '''
        3 варианта.
        False - не решено
        True - Решено
        Int - найдены ошибки (количество квыдратов, в которых ни одного элемента нельзы поставить).
        '''
        num_ambiguous = 0
        num_errors = 0
        for y in range(9):
            for x in range(9):
                one, idx = self.is_only_one(x,y)
                if idx<0:
                    num_errors += 1
                if not one:
                    num_ambiguous += 1
        if num_errors==0 and num_ambiguous==0:
            return True
        if num_errors!=0:
            return num_errors
        return False
    
    def _make_sequaences(self):
        ''' добавляем последовательности - строчки, столбцы, box '''
        self.sequences_row = []
        for y in range(9):
            self.sequences_row.append([(x,y) for x in range(9)])

        self.sequences_col = []
        for x in range(9):
            self.sequences_col.append([(x,y) for y in range(9)])

        self.sequences_box = []
        for y in range(0,9,3):
            for x in range(0,9,3):
                self.sequences_box.append([(x + i%3, y + i//3) for i in range(9)])
        self.sequences = self.sequences_row + self.sequences_col + self.sequences_box

    def in_box(self, x, y, ibox):
        #ibox - индекс последовательности в sequences_box
        xbox = (ibox%3)*3
        ybox = (ibox//3)*3
        return x>=xbox and x<xbox+3 and y>=ybox and y<ybox+3

    def _filter_hints(self):
        ''' Самым тривиальным образом удаляем горизонтальные, вертикальные и box числа, которые не могут быть в этой ячейке. '''
        cleared = False
        for y in range(9):
            for x in range(9):
                if x==8 and y==0:
                    pass

                mask = self.hints[y,x,:]
                
                #убираем из маски все горизонтальные числа
                for xi in range(9):
                    if x != xi:
                        one, one_idx = self.is_only_one(xi, y)
                        if one:
                            if mask[one_idx]:
                                cleared = True
                            mask[one_idx] = 0

                #убираем из маски все вертикальные числа
                for yi in range(9):
                    if y != yi:
                        one, one_idx = self.is_only_one(x, yi)
                        if one:
                            if mask[one_idx]:
                                cleared = True
                            mask[one_idx] = 0

                #убираем из маски все box числа
                xi = (x//3)*3
                yi = (y//3)*3
                for i in range(9):
                    xx = xi + i%3
                    yy = yi + i//3
                    if not(x==xx and y==yy):
                        one, one_idx = self.is_only_one(xx, yy)
                        if one_idx<0:
                            #assert 0
                            continue
                        if one:
                            if mask[one_idx]:
                                cleared = True
                            mask[one_idx] = 0

        print("_filter_hints", cleared)
        return cleared

    def _filter_hints2_0(self):
        #ищем в последовательностях, число, которое есть только в одной ячейке.
        cleared = False
        for sequence in self.sequences:
            count = [0]*9
            last = [None]*9
            for x,y in sequence:
                cell_hints = self.hints[y,x,:]
                for i in range(9):
                    if cell_hints[i]:
                        count[i] += 1
                        last[i] = (x,y)
            for i,c in enumerate(count):
                if c==1:
                    #значит это число в одном экземпляре и его можно переписать
                    x,y = last[i]
                    one,_ = self.is_only_one(x,y)
                    if not one:
                        cleared = True
                    cell_hints = self.hints[y,x,:]
                    cell_hints[:] = 0
                    cell_hints[i] = 1
        print("_filter_hints2_0", cleared)
        return cleared

    def _filter_hints2(self):
        #Ищем в строчках, столбцах и box двойки чисел одинаковые. Если находим, то значит можно отфильтровать остальную линию.
        cleared = False
        for sequence in self.sequences:
            doubles = []
            for x,y in sequence:
                db = self.is_double(x,y)
                if not (db is None):
                    #doubles[z][0] = (x,y)
                    #doubles[z][1] = дублирующиеся в двух ячейках цифры
                    doubles.append(((x,y),db))
            found = None
            for i in range(len(doubles)):
                first = doubles[i]
                for j in range(i+1, len(doubles)):
                    second = doubles[j]
                    if first[1]==second[1]:
                        #дубль найден, фильтруем остальные элементы
                        found = True
                        break
                if found:
                    break
            
            if found:
                excludes = first[1]
                for x,y in sequence:
                    if (x,y)==first[0]:
                        continue
                    if (x,y)==second[0]:
                        continue
                    if self.hints[y,x,excludes[0]]:
                        cleared = True
                    self.hints[y,x,excludes[0]] = 0

                    if self.hints[y,x,excludes[1]]:
                        cleared = True
                    self.hints[y,x,excludes[1]] = 0
                pass

            print("_filter_hints2", cleared)
            return cleared

    def _filter_hints3(self):
        '''Если числа в box находятся на только в одном из столбцов/строк
        то можно их вычистить из других box на этих строках/столбцах.
        '''
        #row, col - box в котором встречалось число
        #-1 - число не встречалось
        #-2 - число встретилось в нескольких box
        #второй индекс числа 1..9
        #первый индекс для row это y
        #первый индекс для col это x
        cleared = False
        for ibox, sequence in enumerate(self.sequences_box):
            row = np.zeros((9,), dtype=np.int8)
            col = np.zeros((9,), dtype=np.int8)
            row[:] = -1
            col[:] = -1
            startx, starty = sequence[0]
            for x,y in sequence:
                is_one, _ = self.is_only_one(x,y)
                if is_one:#не обязательное условие, для теста
                    continue
                cell_hints = self.hints[y,x,:]
                for i in range(9):
                    if cell_hints[i]:
                        if row[i]==-1:
                            row[i] = x
                        elif row[i]!=x:
                            row[i] = -2
                        if col[i]==y:
                            col[i] = y
                        elif col[i]!=y:
                            col[i] = -2

            #очищаем найденные числа в других box
            #так как функция нестабильно, выходим после первой очистки
            for i in range(row.shape[0]):
                xs = row[i]
                if xs >=0:
                    for x,y in self.sequences_col[xs]:
                        if not self.in_box(x,y, ibox):
                            if self.hints[y,x,i]:
                                cleared = True
                            self.hints[y,x,i] = 0

            for i in range(row.shape[0]):
                ys = col[i]
                if ys >=0:
                    for x,y in self.sequences_row[ys]:
                        if not self.in_box(x,y, ibox):
                            if self.hints[y,x,i]:
                                cleared = True
                            self.hints[y,x,i] = 0

            if cleared:
                print("_filter_hints3", True)
                return True

        print("_filter_hints3", False)
        return False
