#pragma once
#include <string>
#include <stdint.h>

/*
    Судоку - это строчка на 81 char
*/

std::string sudoku_solve(const std::string& initial_data, bool print_solve_time=false);

void sudoku_print(const std::string& data);