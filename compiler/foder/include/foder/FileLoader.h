/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <fstream>
#include <vector>

namespace foder
{

class FileLoader
{
private:
  using DataBuffer = std::vector<char>;

public:
  explicit FileLoader(const std::string &path) : _path(path) {}

public:
  FileLoader(const FileLoader &) = delete;
  FileLoader(FileLoader &&) = delete;

public:
  const DataBuffer load(void)
  {
    std::ifstream file(_path, std::ios::binary | std::ios::in);
    if (!file.good())
      throw std::runtime_error("Couldn't open file.");

    file.unsetf(std::ios::skipws);

    file.seekg(0, std::ios::end);
    _size = file.tellg();
    file.seekg(0, std::ios::beg);

    // reserve capacity
    DataBuffer data(_size);

    // read the data
    file.read(data.data(), _size);
    if (file.fail())
      throw std::runtime_error("Couldn't read file.");

    return data;
  }

  inline std::streampos size(void) { return _size; }

private:
  const std::string _path;
  std::streampos _size;
};

} // namespace foder
