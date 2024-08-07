
def get_puzzle(filename="data/puzzles0_kaggle", idx=None):
    #idx==None - берём случайный пазл
    #idx==5 - берём 5-ый пазл
    #idx=='all' - возвращаем list паззлов вместе
    with open(filename, "rb") as f:
        if idx=='all':
            lines = []
            while True:
                line = f.readline()
                if len(line)==0:
                    break
                if line[0]==b'#'[0]:
                    continue
                line = line.strip()
                assert len(line)==81
                lines.append(line)
            return lines

        while True:
            line = f.readline()
            if line[0]!=b'#'[0]:
                break
        if idx is None:
            import random
            ri = random.randint(0, 300)
        else:
            ri = idx
        print("get_puzzle", filename, ri)
        for _ in range(ri):
            line = f.readline()
        line = line.strip()
        assert len(line)==81
        print(line)
        return line

def get_puzzle_csv(filename="/home/balmer/radio/ML/sudoku/kaggle_1million/sudoku.csv", max_count=-1):
    quizzes = []
    with open(filename, 'r') as file:
        file.readline()

        if max_count>0:
            for line in file:
                quiz, solution = line.split(",")
                quizzes.append(quiz.encode())
                max_count -= 1
                if max_count==0:
                    break
        else:
            for line in file:
                quiz, solution = line.split(",")
                quizzes.append(quiz.encode())

    return quizzes