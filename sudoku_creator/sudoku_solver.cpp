#include "sudoku_solver.h"
#include <string.h>
#include <chrono>
#include "file_utils.h"

SudokuSolver::SudokuSolver() {}
SudokuSolver::~SudokuSolver() {}

int SudokuSolver::is_only_one(uint32_t d)
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

void SudokuSolver::solve()
{
    steps_count = 0;
    recursion_count = 0;
    std::array<bool, 81> empty_mask;
    std::fill( std::begin( empty_mask ), std::end( empty_mask ), false );
    solve_rec(0, empty_mask);
}

bool SudokuSolver::solve_rec(int recursion, const std::array<bool, 81>& recursion_mask)
{
    recursion_count++;
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
            //для начала попробуем тяжелые проверки
            filter_2();
            filter_3();

            if(memcmp(data_prev, data, sizeof(data))==0)
            {
                //Ничего не смогли нарешать, надо рекурсивно выставить какое либо из чисел в одно из валидных.
                int x=0,y=0;
                int bits = 10;
                for(int yy=0; yy<9; yy++)
                {
                    for(int xx=0; xx<9; xx++)
                    {
                        if(recursion_mask[yy*9+xx])
                            continue;

                        int b = bits_count(data[yy][xx]);
                        if(b==1)
                            continue;
                        if(b>=bits)
                            continue;
                        x = xx;
                        y = yy;
                        bits = b;
                        if(bits==2)
                            break;
                    }

                    if(bits==2)
                        break;
                }

                if(bits>=10)
                    return false;

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

                        std::array<bool, 81> recursion_mask1 = recursion_mask;
                        recursion_mask1[y*9+x] = true;
                        if(solve_rec(recursion+1, recursion_mask1))
                            return true;
                        memcpy(data, data_prev, sizeof(data));
                    }
                }
                return false;
            }
        }

        if(print_all)
            print_partial(recursion, steps_count);
    }
    return false;
}

//Берём точно определённые числа, находящихся на этой линии.
//В остальных ячейках этих чисел быть не должно.
void SudokuSolver::filter_h()
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

void SudokuSolver::filter_v()
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

void SudokuSolver::filter_box()
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
void SudokuSolver::filter_uniq()
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
void SudokuSolver::filter_in_one_line_at_box_h()
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

void SudokuSolver::filter_in_one_line_at_box_v()
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

//idx>=0 && idx<9
//Если type==H, то idx-y
//Если type==V, то idx-x
//Если type==BOC, то idx-номер квадрата 3x3
void SudokuSolver::get_line(LType type, int idx, uint32_t out[9])
{
    if(type==LType::H)
    {
        int y = idx;
        for(int x=0; x<9;x++)
            out[x] = data[y][x];
        return;
    }

    if(type==LType::V)
    {
        int x = idx;
        for(int y=0; y<9; y++)
            out[y] = data[y][x];
        return;
    }

    if(type==LType::BOX)
    {
        int box = idx;
        int xb = xbox[box]*3;
        int yb = ybox[box]*3;

        uint32_t mask = all_mask;
        for(int cell=0;cell<9;cell++)
        {
            int x = xb+xbox[cell];
            int y = yb+ybox[cell];
            out[cell] = data[y][x];
        }
    }
}

void SudokuSolver::set_line(LType type, int idx, uint32_t out[9])
{
    if(type==LType::H)
    {
        int y = idx;
        for(int x=0; x<9;x++)
            data[y][x] = out[x];
        return;
    }

    if(type==LType::V)
    {
        int x = idx;
        for(int y=0; y<9; y++)
            data[y][x] = out[y];
        return;
    }

    if(type==LType::BOX)
    {
        int box = idx;
        int xb = xbox[box]*3;
        int yb = ybox[box]*3;

        uint32_t mask = all_mask;
        for(int cell=0;cell<9;cell++)
        {
            int x = xb+xbox[cell];
            int y = yb+ybox[cell];
            data[y][x] = out[cell];
        }
    }
}

int SudokuSolver::bits_count(uint32_t v)
{
    int count = 0;
    for(int j=0; j<9; j++)
        if(v&(1<<j))
            count++;
    return count;
}

//Находим двойки чисел (которые только с двух ячейках) и стираем их все
void SudokuSolver::filter_2()
{
    for(LType ltype=LType::H; ltype<=LType::BOX; ltype = (LType)(1+(int)ltype))
    {
        filter_2(ltype);
    }
}

void SudokuSolver::filter_2(LType ltype)
{
    for(int idx=0; idx<9; idx++)
    {
        uint32_t line[9];
        get_line(ltype, idx, line);
        filter_2(line);
        set_line(ltype, idx, line);
    }
}

void SudokuSolver::filter_2(uint32_t line[9])
{
    int cells[9];
    int cell_count = 0;

    for(int l=0; l<9-1; l++)
    {
        uint32_t vl = line[l];
        if(bits_count(vl)!=2)
            continue;
        //Ищем такой-же
        for(int ll=l+1; ll<9; ll++)
        {
            if(line[ll]==vl)
            {
                //нашли, можно вычищать.
                //Да, успешно нашли, вычищаем
                for(int xl=0; xl<9; xl++)
                {
                    if(line[xl] != vl)
                        line[xl] &= ~vl;
                }
                break;
            }
        }
    }
}

void SudokuSolver::filter_3()
{
    for(LType ltype=LType::H; ltype<=LType::BOX; ltype = (LType)(1+(int)ltype))
    {
        filter_3(ltype);
    }
}

void SudokuSolver::filter_3(LType ltype)
{
    for(int idx=0; idx<9; idx++)
    {
        uint32_t line[9];
        get_line(ltype, idx, line);
        filter_3(line);
        set_line(ltype, idx, line);
    }
}

void SudokuSolver::filter_3(uint32_t line[9])
{
    int cells[9];
    int cell_count = 0;
    int desire_count = 3;

    for(int l=0; l<9-1; l++)
    {
        uint32_t vl = line[l];
        if(bits_count(vl)!=desire_count)
            continue;
        //Ищем такой-же
        int equal_count = 0;
        for(int ll=l+1; ll<9; ll++)
            if(line[ll]==vl)
                equal_count++;

        if(equal_count==desire_count-1)
        {
            //нашли, можно вычищать.
            //Да, успешно нашли, вычищаем
            for(int xl=0; xl<9; xl++)
            {
                if(line[xl] != vl)
                    line[xl] &= ~vl;
            }
            break;
        }
    }
}

std::string sudoku_solve(const std::string& initial_data, bool print_solve_time, bool print_all)
{
    SudokuSolver s;
    s.print_all = print_all;
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
        printf("Solve time=%.1f us steps=%i recursion=%i\n", nano_seconds*1e-3, s.steps_count, s.recursion_count);
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
    print_partial();
}

bool SudokuSolver::print_partial(const std::string& filename)
{
    FILE* f = fopen(filename.c_str(), "wb");
    if(f==NULL)
    {
        fprintf(stderr, "Cannot open file %s\n", filename.c_str());
        return false;
    }

    print_partial(f, false);
    fclose(f);
    return true;
}

void SudokuSolver::print_partial(FILE* f, bool pretty)
{
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

            if(pretty)
            {
                int o = is_only_one(x, y);
                if(o!=-1)
                {
                    str[y*3+1][x*4+1] = o>0?o+'0':'X';
                    continue;
                }
            }

            uint32_t v = data[y][x];
            if(pretty)
            {
                if(v==all_mask)
                    continue;
            }

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
        fprintf(f, "%s\n", str[y]);
        if(y%3==2)
        {
            if(y==2*3+2 || y==5*3+2)
                fprintf(f, "--- --- --- --- --- --- --- --- ---\n");
            else
                fprintf(f, "\n");
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

void SudokuSolver::clear(uint32_t mask)
{
    for(int y=0;y<9;y++)
        for(int x=0;x<9;x++)
            data[y][x] = mask;
}

bool SudokuSolver::load(const std::string& filename)
{
    std::vector<uint8_t> file_data;
    if(!read_file(filename.c_str(), file_data))
    {
        fprintf(stderr, "Cannot read file %s\n", filename.c_str());
        return false;
    }

    clear(0);

    int line_num = 0;
    for(size_t start_line=0; start_line<file_data.size();)
    {
        size_t end_line = start_line;
        for(;end_line<file_data.size();end_line++)
        {
            if(file_data[end_line]=='\n')
                break;
        }

        if(!(end_line-start_line==35 || (end_line-start_line==0 && line_num%4==3)))
        {
            fprintf(stderr, "Line %i shuld be 36 characters", line_num+1);
            return false;
        }

        int y = line_num/4;
        int sub_line = line_num%4;
        if(sub_line!=3)
        {
            for(int x=0; x<9; x++)
            {
                for(int sx=0; sx<3; sx++)
                {
                    char c = file_data[start_line + x*4 + sx];
                    if(c=='.')
                        continue;
                    if(!(c>='1' && c<='9'))
                    {
                        fprintf(stderr, "Unknown character %c", c);
                        return false;
                    }

                    int j = c-'1';
                    data[y][x] |= 1<<j;
                }
            }
        }

        start_line = end_line+1;
        line_num++;
        if(line_num==36)
            break;
    }

    return true;
}