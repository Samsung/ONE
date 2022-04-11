/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "trix_loader.h"

#include <fcntl.h>    // O_RDONLY
#include <sys/stat.h> // fstat
#include <unistd.h>   // close

namespace onert
{
namespace trix_loader
{

namespace
{
class Verifier
{
public:
  Verifier(const std::uint8_t *buf, size_t buf_len) : _buf(buf), _buf_len(buf_len) {}
  bool verify() const
  {
    (void)_buf;
    (void)_buf_len;
    return true;
  }

private:
  const uint8_t *_buf;
  size_t _buf_len;
};
} // namespace

void TrixLoaderBase::loadFromFile(const std::string &file_path)
{
  (void)_subgraphs; // make android toolchain happy from unused member warning
  _fd = open(file_path.c_str(), O_RDONLY);
  if (_fd < 0)
  {
    throw std::runtime_error("Failed to open file " + file_path);
  }

  struct stat file_stat;
  if (fstat(_fd, &file_stat) != 0)
  {
    throw std::runtime_error("Fstat failed or file " + file_path + " is not a regular file");
  }
  _sz = file_stat.st_size;

  // Map model file into memory region
  _base = static_cast<uint8_t *>(mmap(NULL, _sz, PROT_READ, MAP_PRIVATE, _fd, 0));
  if (_base == MAP_FAILED)
  {
    close(_fd);
    throw std::runtime_error("mmap failed - " + std::string(strerror(errno)));
  }

  verifyModel();
  loadModel();
  munmap(_base, _sz);

  close(_fd);
  throw std::runtime_error("Not implemented yet");
}

} // namespace trix_loader
} // namespace onert
