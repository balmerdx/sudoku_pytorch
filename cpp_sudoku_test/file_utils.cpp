#include "file_utils.h"

#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#ifdef __APPLE__
#define lseek64 lseek
typedef off_t off64_t;
#endif

bool read_file(const char* path, std::vector<uint8_t>& data)
{
    data.clear();
    int fd = open (path, O_RDONLY);
    if(fd==-1)
        return false;
    off64_t file_size = lseek64 (fd, 0, SEEK_END);
    if(file_size < 0)
        return false;
    lseek64 (fd, 0, SEEK_SET);
    data.resize(file_size);
    ssize_t ret = read(fd, data.data(), file_size);

    close(fd);
    return ret == file_size;
}

bool write_file(const char* path, const std::vector<uint8_t>& data)
{
    int fd = open (path, O_WRONLY | O_CREAT | O_TRUNC, S_IWUSR | S_IRUSR);
    if(fd==-1)
        return false;
    ssize_t ret = write(fd, data.data(), data.size());
    close(fd);
    return ret == data.size();
}

bool write_file_if_changed(const char* path, const std::vector<uint8_t>& data)
{
    std::vector<uint8_t> data1;
    if(!read_file(path, data1))
        return write_file(path, data);

    if(data==data1)
        return true;

    return write_file(path, data);
}
