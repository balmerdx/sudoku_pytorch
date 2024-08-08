#include "sudoku_solver.h"
#include <string.h>
#include <chrono>

class SudokuSolver
{
public:
    static const uint32_t all_mask = 0x1FF;
    static constexpr int xbox[9]={0,1,2,0,1,2,0,1,2};
    static constexpr int ybox[9]={0,0,0,1,1,1,2,2,2};
    int steps_count;
    bool print_all = false;
public:
    //У нас битовая маска.
    //9 бит. 1 - цифра возможна здесь. 0 - невозможна
    uint32_t data[9][9];

    SudokuSolver() {}
    ~SudokuSolver() {}

    bool init(const std::string& initial_data);
    std::string to_string();

    //Частичное решение
    void print_partial(int recursion, int iteration);
    void print_full();

    //return -1 если непонятно какое число
    //return 0 если никакого числа
    //return 1..9 если число определённое
    int is_only_one(int x, int y)
    {
        return is_only_one(data[y][x]);
    }

    int is_only_one(uint32_t d)
    {
        if(d==0)
            return 0;
        for(int i=0;i<9;i++)
        {
            uint32_t v = 1<<i;
            if(d&v)
            {
                if(d==v)
                    return i+1;
                else
                    return -1;
            }
        }
        return 0;
    }

    void solve()
    {
        steps_count = 0;
        solve_rec(0, 0, 0);
    }

    bool solve_rec(int x, int y, int recursion)
    {
        while(true)
        {
            steps_count++;
            uint32_t data_prev[9][9];
            memcpy(data_prev, data, sizeof(data));
            filter_box();
            //print_partial(recursion, steps_count);
            filter_uniq();
            filter_in_one_line_at_box_h();
            filter_in_one_line_at_box_v();

            //Проверяем решение на валидность или ошибочность.
            bool all_determined = true;
            for(int y=0;y<9;y++)
            {
                uint32_t mask = all_mask;
                for(int x=0;x<9;x++)
                {
                    int v = is_only_one(x,y);
                    if(v==0)
                    {
                        if(print_all)
                            print_partial(recursion, steps_count);
                        return false;
                    }
                    if(v<0)
                        all_determined = false;
                }
            }

            if(all_determined)
            {
                if(print_all)
                {
                    printf("Solved!\n");
                    print_partial(recursion, steps_count);
                }
                return true;
            }

            if(memcmp(data_prev, data, sizeof(data))==0)
            {
                //Ничего не смогли нарешать, надо рекурсивно выставить какое либо из чисел в одно из валидных.
                while(true)
                {
                    int v = is_only_one(x, y);
                    if(v<0)
                        break;
                    x++;
                    if(x==9)
                    {
                        x = 0;
                        y++;
                    }

                    if(!(x<9 && y<9))
                        return false;
                }

                memcpy(data_prev, data, sizeof(data));
                auto v = data[y][x];
                for(int j=0; j<9;j++)
                {
                    if(v&(1<<j))
                    {
                        data[y][x] = 1<<j;
                        if(print_all)
                        {
                            printf("Try %i at (%i,%i) ", j+1, x,y);
                            print_partial(recursion, steps_count);
                        }
                        if(solve_rec(x, y, recursion+1))
                            return true;
                        memcpy(data, data_prev, sizeof(data));
                    }
                }
                return false;
            }
            if(print_all)
                print_partial(recursion, steps_count);
        }
        return false;
    }

    //Берём точно определённые числа, находящихся на этой линии.
    //В остальных ячейках этих чисел быть не должно.
    void filter_h()
    {
        for(int y=0;y<9;y++)
        {
            uint32_t mask = all_mask;
            for(int x=0;x<9;x++)
            {
                int v = is_only_one(x,y);
                if(v>0)
                    mask &= ~(1<<(v-1));
            }

            for(int x=0;x<9;x++)
            {
                if(is_only_one(x,y)>0)
                    continue;
                data[y][x] &= mask;
            }
        }
    }

    void filter_v()
    {
        for(int x=0;x<9;x++)
        {
            uint32_t mask = all_mask;
            for(int y=0;y<9;y++)
            {
                int v = is_only_one(x,y);
                if(v>0)
                    mask &= ~(1<<(v-1));
            }

            for(int y=0;y<9;y++)
            {
                if(is_only_one(x,y)>0)
                    continue;
                data[y][x] &= mask;
            }
        }
    }

    void filter_box()
    {
        int one[9][9];
        for(int y=0;y<9;y++)
            for(int x=0;x<9;x++)
                one[y][x] = is_only_one(x,y);


        for(int box=0;box<9;box++)
        {
            int xb = xbox[box]*3;
            int yb = ybox[box]*3;

            uint32_t mask = all_mask;
            for(int cell=0;cell<9;cell++)
            {
                int x = xb+xbox[cell];
                int y = yb+ybox[cell];
                int v = one[y][x];
                if(v>0)
                    mask &= ~(1<<(v-1));
            }

            for(int cell=0;cell<9;cell++)
            {
                int x = xb+xbox[cell];
                int y = yb+ybox[cell];
                if(one[y][x]>0)
                    continue;
                data[y][x] &= mask;
            }
        }
    }

    //Элемент находится только в одной из ячеек в h/v/box
    void filter_uniq()
    {
        //uniq h
        for(int y=0;y<9;y++)
        {
            int count[9] = {};
            for(int x=0;x<9;x++)
            {
                auto v = data[y][x];
                for(int j=0; j<9; j++)
                    if(v&(1<<j))
                        count[j]++;
            }

            for(int j=0;j<9;j++)
            if(count[j]==1)
            {
                for(int x=0;x<9;x++)
                {
                    auto& v = data[y][x];
                    if(v&(1<<j) & v!=(1<<j))
                        v = 1<<j;
                }
            }
        }

        //uniq v
        for(int x=0;x<9;x++)
        {
            int count[9] = {};
            for(int y=0;y<9;y++)
            {
                auto v = data[y][x];
                for(int j=0; j<9; j++)
                    if(v&(1<<j))
                        count[j]++;
            }

            for(int j=0;j<9;j++)
            if(count[j]==1)
            {
                for(int y=0;y<9;y++)
                {
                    auto& v = data[y][x];
                    if(v&(1<<j) & v!=(1<<j))
                        v = 1<<j;
                }
            }
        }

        //uniq box
        for(int box=0;box<9;box++)
        {
            int xb = xbox[box]*3;
            int yb = ybox[box]*3;

            int count[9] = {};

            uint32_t mask = all_mask;
            for(int cell=0;cell<9;cell++)
            {
                int x = xb+xbox[cell];
                int y = yb+ybox[cell];
                auto v = data[y][x];
                for(int j=0; j<9; j++)
                    if(v&(1<<j))
                        count[j]++;
            }
            for(int j=0;j<9;j++)
            if(count[j]==1)
            {
                for(int cell=0;cell<9;cell++)
                {
                    int x = xb+xbox[cell];
                    int y = yb+ybox[cell];
                    auto& v = data[y][x];
                    if(v&(1<<j) & v!=(1<<j))
                        v = 1<<j;
                }
            }
        }
    }

    /*
Если число в box находится исключительно в одном из horizontal, vertical кусков box, то значит его можно вычищать из остальных box.
Например мы хотим это для горизонтального случая сделать.
Т.е. нам надо будет пройтись по всем трём строкам и определить какие числа там находятся.
Потом у нас будет условие - что число находится в одной строке, но его нет в других.
Имея такую маску мы можем её инвертировать и пройтись по другим box в этой строке.
    */
    void filter_in_one_line_at_box_h()
    {
        for(int yb=0;yb<3;yb++)
        {
            uint32_t mask[3][3] = {};
            for(int xb=0;xb<3;xb++)
                for(int yj=0;yj<3;yj++)
                    for(int xj=0;xj<3;xj++)
                        mask[xb][yj] |= data[yb*3+yj][xb*3+xj];
            uint32_t mask_uniq[3][3] = {};
            for(int xb=0;xb<3;xb++)
                for(int yj=0;yj<3;yj++)
                {
                    mask_uniq[xb][yj] = mask[xb][yj];
                    for(int yi=0;yi<3;yi++)
                        if(yi!=yj)
                            mask_uniq[xb][yj] &= ~mask[xb][yi];
                }

            for(int xb=0;xb<3;xb++)
                for(int yj=0;yj<3;yj++)
                    for(int xj=0;xj<3;xj++)
                        for(int xc=0;xc<3;xc++)
                            if(xb!=xc)
                                data[yb*3+yj][xb*3+xj] &= ~mask_uniq[xc][yj];
        }
    }

    void filter_in_one_line_at_box_v()
    {
        for(int xb=0;xb<3;xb++)
        {
            uint32_t mask[3][3] = {};
            for(int yb=0;yb<3;yb++)
                for(int xj=0;xj<3;xj++)
                    for(int yj=0;yj<3;yj++)
                        mask[yb][xj] |= data[yb*3+yj][xb*3+xj];
            uint32_t mask_uniq[3][3] = {};
            for(int yb=0;yb<3;yb++)
                for(int xj=0;xj<3;xj++)
                {
                    mask_uniq[yb][xj] = mask[yb][xj];
                    for(int xi=0;xi<3;xi++)
                        if(xi!=xj)
                            mask_uniq[yb][xj] &= ~mask[yb][xi];
                }

            for(int yb=0;yb<3;yb++)
                for(int xj=0;xj<3;xj++)
                    for(int yj=0;yj<3;yj++)
                        for(int yc=0;yc<3;yc++)
                            if(yb!=yc)
                                data[yb*3+yj][xb*3+xj] &= ~mask_uniq[yc][xj];
        }
    }
};


std::string sudoku_solve(const std::string& initial_data, bool print_solve_time)
{
    SudokuSolver s;
    if(!s.init(initial_data))
        return std::string();

    std::chrono::steady_clock::time_point start, stop;
    if(print_solve_time)
        start = std::chrono::steady_clock::now();

    s.solve();

    if(print_solve_time)
    {
        stop = std::chrono::steady_clock::now();
        double nano_seconds = (stop-start).count();
        printf("Solve time=%.1f us steps=%i\n", nano_seconds*1e-3, s.steps_count);
    }

    return s.to_string();
}

void sudoku_print(const std::string& data)
{
    SudokuSolver s;
    if(!s.init(data))
        return;
    s.print_full();
}

bool SudokuSolver::init(const std::string& initial_data)
{
    if(initial_data.size()!=81)
        return false;
    
    for(int y=0;y<9;y++)
    for(int x=0;x<9;x++)
    {
        int c = initial_data[x + y*9];
        c = c - '1';
        uint32_t v = all_mask;
        if(c>=0 && c<9)
            v = 1<<c;
        data[y][x] = v;
    }

    return true;
}

std::string SudokuSolver::to_string()
{
    std::string s(81,'.');
    for(int y=0;y<9;y++)
    for(int x=0;x<9;x++)
    {
        int v = is_only_one(x,y);
        if(v>0)
            s[x + y*9] = v+'0';
        if(v==0)
            s[x + y*9] = 'X';
    }


    return s;
}

void SudokuSolver::print_partial(int recursion, int iteration)
{
    printf("R:%i It:%i\n", recursion, iteration);

    char str[9*3][9*4];

    for(int y=0;y<9;y++)
    {
        for(int x=0;x<9;x++)
        {
            for(int yy=0;yy<3;yy++)
            {
                char* c = &str[y*3+yy][x*4];
                c[0]='.';
                c[1]='.';
                c[2]='.';
                char r = ' ';
                if(x==8)
                    r=0;
                if(x==2 || x==5)
                    r='|';
                c[3]=r;
            }

            int o = is_only_one(x, y);
            if(o!=-1)
            {
                str[y*3+1][x*4+1] = o>0?o+'0':'X';
                continue;
            }

            uint32_t v = data[y][x];
            if(v==all_mask)
                continue;
            for(int j=0;j<9;j++)
            {
                int yy=j/3;
                int xx=j%3;
                if(v&(1<<j))
                    str[y*3+yy][x*4+xx] = j+'1';
            }

        }    
    }

    for(int y=0;y<9*3;y++)
    {
        printf("%s\n", str[y]);
        if(y%3==2)
        {
            if(y==2*3+2 || y==5*3+2)
                printf("--- --- --- --- --- --- --- --- ---\n");
            else
                printf("\n");
        }
    }
}

void SudokuSolver::print_full()
{
    for(int y=0;y<9;y++)
    {
        for(int x=0;x<9;x++)
        {
            int v = is_only_one(x, y);
            if(v<=0)
                printf(v==0?"X":".");
            else
                printf("%i", v);
        }    
        printf("\n");
    }
}