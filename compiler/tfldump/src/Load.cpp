/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <tflread/Model.h>

#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>

namespace
{

class MemoryMappedModel final : public tflread::Model
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
  const ::tflite::Model *model(void) const override { return ::tflite::GetModel(_data); }

private:
  int _fd = -1;
  void *_data = nullptr;
  size_t _size = 0;
};

class FileDescriptor final
{
public:
  FileDescriptor(int value) : _value{value}
  {
    // DO NOTHING
  }

public:
  // NOTE Copy is not allowed
  FileDescriptor(const FileDescriptor &) = delete;

public:
  // NOTE Move is allowed
  FileDescriptor(FileDescriptor &&fd) { _value = fd.release(); }

public:
  ~FileDescriptor()
  {
    if (_value != -1)
    {
      // Close on destructor
      close(_value);
    }
  }

public:
  int value(void) const { return _value; }

public:
  int release(void)
  {
    auto res = _value;
    _value = -1;
    return res;
  }

private:
  int _value = -1;
};

} // namespace

namespace tflread
{

std::unique_ptr<Model> load_tflite(const std::string &path)
{
  FileDescriptor fd = open(path.c_str(), O_RDONLY);

  if (fd.value() == -1)
  {
    // Return nullptr on open failure
    return nullptr;
  }

  struct stat st;
  if (fstat(fd.value(), &st) == -1)
  {
    // Return nullptr on fstat failure
    return nullptr;
  }

  auto size = st.st_size;
  auto data = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd.value(), 0);

  if (data == MAP_FAILED)
  {
    // Return nullptr on mmap failure
    return nullptr;
  }

  return std::unique_ptr<tflread::Model>{new MemoryMappedModel(fd.release(), data, size)};
}

} // namespace tflread
