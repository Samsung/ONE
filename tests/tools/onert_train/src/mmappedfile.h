/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __ONERT_TRAIN_MMAPPED_FILE_H__
#define __ONERT_TRAIN_MMAPPED_FILE_H__

#include <fcntl.h>    // O_RDONLY
#include <sys/mman.h> // mmap, munmap
#include <sys/stat.h> // fstat
#include <unistd.h>   // close
#include <stdint.h>   // SIZE_MAX

namespace onert_train
{

class MMappedFile
{
public:
  MMappedFile(const char *filename) { _fd = open(filename, O_RDWR); }
  ~MMappedFile() { close(); }

  bool ensure_mmap()
  {
    struct stat file_stat;
    if (fstat(_fd, &file_stat) != 0 || file_stat.st_size < 0 ||
        static_cast<uint64_t>(file_stat.st_size) > SIZE_MAX)
      return false;

    _buf_sz = static_cast<size_t>(file_stat.st_size);
    _buf = mmap(NULL, _buf_sz, PROT_READ | PROT_WRITE, MAP_SHARED, _fd, 0);
    return _buf != MAP_FAILED;
  }

  bool sync() { return msync(_buf, _buf_sz, MS_SYNC) == 0; }

  bool close()
  {
    bool ret = false;
    if (_buf != MAP_FAILED)
    {
      ret = munmap(_buf, _buf_sz) == 0;
      _buf = MAP_FAILED; // mark as cleaned up
    }
    if (_fd != -1)
    {
      ::close(_fd);
      _fd = -1; // mark as cleaned up
    }
    return ret;
  }

  uint8_t *buf() const { return static_cast<uint8_t *>(_buf); }
  size_t buf_size() const { return _buf_sz; }

private:
  int _fd;
  void *_buf = MAP_FAILED;
  size_t _buf_sz = 0;
};

} // namespace onert_train

#endif // __ONERT_TRAIN_MMAPPED_FILE_H__
