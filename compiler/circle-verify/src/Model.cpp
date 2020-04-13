/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Model.h"

#include <cwrap/Fildes.h>

#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>

namespace
{

class MemoryMappedModel final : public ModelData
{
public:
  /**
   * @require fd and data SHOULD be valid
   */
  explicit MemoryMappedModel(int fd, void *data, size_t size) : _fd{fd}, _data{data}, _size{size}
  {
    // DO NOTHING
  }

public:
  ~MemoryMappedModel()
  {
    munmap(_data, _size);
    close(_fd);
  }

public:
  MemoryMappedModel(const MemoryMappedModel &) = delete;
  MemoryMappedModel(MemoryMappedModel &&) = delete;

public:
  const void *data(void) const override { return _data; };
  const size_t size(void) const override { return _size; };

private:
  int _fd = -1;
  void *_data = nullptr;
  size_t _size = 0;
};

} // namespace

std::unique_ptr<ModelData> load_modeldata(const std::string &path)
{
  cwrap::Fildes fd(open(path.c_str(), O_RDONLY));

  if (fd.get() == -1)
  {
    // Return nullptr on open failure
    return nullptr;
  }

  struct stat st;
  if (fstat(fd.get(), &st) == -1)
  {
    // Return nullptr on fstat failure
    return nullptr;
  }

  auto size = st.st_size;
  auto data = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd.get(), 0);

  if (data == MAP_FAILED)
  {
    // Return nullptr on mmap failure
    return nullptr;
  }

  return std::unique_ptr<ModelData>{new MemoryMappedModel(fd.release(), data, size)};
}
