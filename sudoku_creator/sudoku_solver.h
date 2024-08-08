#pragma once
#include <string>
#include <stdint.h>
#include <array>

/*
    Судоку - это строчка на 81 char
*/

std::string sudoku_solve(const std::string& initial_data, bool print_solve_time=false, bool print_all=false);
void sudoku_print(const std::string& data);
void sudoku_simply_test(const std::string& filename);


//internal interface
class SudokuSolver
{
public:
    static const uint32_t all_mask = 0x1FF;
    static constexpr int xbox[9]={0,1,2,0,1,2,0,1,2};
    static constexpr int ybox[9]={0,0,0,1,1,1,2,2,2};
    int steps_count;
    bool print_all = false;

    enum class LType
    {
        //не менять местами
        H,//Horizontal
        V,//Vertical
        BOX
    };

public:
    //У нас битовая маска.
    //9 бит. 1 - цифра возможна здесь. 0 - невозможна
    uint32_t data[9][9];

    SudokuSolver();
    ~SudokuSolver();

    bool init(const std::string& initial_data);
    bool load(const std::string& filename);
    void clear(uint32_t mask=0);
    std::string to_string();

    //Частичное решение
    void print_partial(int recursion, int iteration);
    void print_partial(FILE* f=stdout, bool pretty=true);
    bool print_partial(const std::string& filename);
    void print_full();

    //return -1 если непонятно какое число
    //return 0 если никакого числа
    //return 1..9 если число определённое
    int is_only_one(int x, int y) { return is_only_one(data[y][x]); }
    int is_only_one(uint32_t d);

    void solve();
    bool solve_rec(int recursion, const std::array<bool, 81>& recursion_mask);

    //Берём точно определённые числа, находящихся на этой линии.
    //В остальных ячейках этих чисел быть не должно.
    void filter_h();
    void filter_v();
    void filter_box();
    //Элемент находится только в одной из ячеек в h/v/box
    void filter_uniq();

    /*
Если число в box находится исключительно в одном из horizontal, vertical кусков box, то значит его можно вычищать из остальных box.
Например мы хотим это для горизонтального случая сделать.
Т.е. нам надо будет пройтись по всем трём строкам и определить какие числа там находятся.
Потом у нас будет условие - что число находится в одной строке, но его нет в других.
Имея такую маску мы можем её инвертировать и пройтись по другим box в этой строке.
    */
    void filter_in_one_line_at_box_h();
    void filter_in_one_line_at_box_v();

    //idx>=0 && idx<9
    //Если type==H, то idx-y
    //Если type==V, то idx-x
    //Если type==BOC, то idx-номер квадрата 3x3
    void get_line(LType type, int idx, uint32_t out[9]);
    void set_line(LType type, int idx, uint32_t out[9]);
    int bits_count(uint32_t v);

    //Находим двойки чисел (которые только с двух ячейках) и стираем их все
    void filter_2();
    void filter_2(LType ltype);
    void filter_2(uint32_t line[9]);

    void filter_3();
    void filter_3(LType ltype);
    void filter_3(uint32_t line[9]);
};
