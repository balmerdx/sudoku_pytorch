#include <stdio.h>
#include "sudoku_solver.h"

void test()
{
    std::string sudoku(81, '.');
    //std::string sudoku = ".................1.....2.3......3.2...1.4......5....6..3......4.7..8...962...7...";
    SudokuSolver s;
    //s.clear(s.all_mask);s.print_partial("../test_data/all.txt");return;

    if(true)
    {
        s.load("../test_data/test_filter_2.txt");
        s.filter_2();
        s.print_partial(0,0);
    }

    if(false)
    {
        s.load("../test_data/test_filter_3.txt");
        s.filter_3();
        s.print_partial(0,0);
    }
    //
}

int main(int argc, char** argv)
{
    //test(); return 0;
    //std::string sudoku = "9...84.6.6.4..52.7.3..7..8.76...15...53.....1...4.96.31.5.26.9...2.4....8....371."; //easy
    std::string sudoku = "5..6....3.....3.2.9..2.7.8...8....32..43..9.5..9....688..7.2.5......4.7.7..9....6"; //veryhard
    //std::string sudoku = "8.........95.......76.........426798...571243...893165......916....3.487....1.532"; //puzzles7_serg_benchmark extra hard
    //std::string sudoku = "4...3.......6..8..........1....5..9..8....6...7.2........1.27..5.3....4.9........";//magictour 0
    //std::string sudoku = ".................1.....2.3......3.2...1.4......5....6..3......4.7..8...962...7...";//data/puzzles2_17_clue 0
    //std::string sudoku = ".................1.....2.34.....4.....5...6....6.3.....3..6.....7..5.8..24......7";//data/puzzles2_17_clue 19
    //std::string sudoku = "..3.7..4...6..23.1.89.........1.7.8.517.....6...4.....271..9..5.95..........2...."; //puzzles1_unbiased 0
    //std::string sudoku = "7..51...3..8...7.....4.......6....9.3...7.2...8...4..1.....26.........4.5.9.8...."; //tdoku build/generate -p0 -c0 -g1 -d1 -n100 -e50 -s0
    std::string solved = sudoku_solve(sudoku, true, false);
    printf("%s\n", sudoku.c_str());
    printf("%s\n", solved.c_str());
    sudoku_print(solved);
    return 0;
}