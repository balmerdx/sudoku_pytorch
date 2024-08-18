cmake_minimum_required(VERSION 3.0.0)
project(sudoku_creator VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(${CMAKE_PROJECT_NAME}
    main.cpp
    sudoku_solver.cpp
    file_utils.cpp
    )

