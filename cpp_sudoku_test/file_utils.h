#pragma once
#include <stdint.h>
#include <vector>

bool read_file(const char* path, std::vector<uint8_t>& data);
bool write_file(const char* path, const std::vector<uint8_t>& data);

//Cравнивает data со считанным посредством read_file
//Если они не совпадают, то пишет данные на SSD
bool write_file_if_changed(const char* path, const std::vector<uint8_t>& data);
