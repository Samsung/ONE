/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "crew/PConfigIni.h"

#include <foder/FileLoader.h>

#include <cassert>
#include <cstring>
#include <stdexcept>
#include <string>

namespace crew
{

Sections read_ini(const char *data, size_t length)
{
  assert(data != nullptr);
  assert(length > 0);

  auto buffer = std::vector<char>();
  buffer.reserve(length + 1);
  char *pbuffer = buffer.data();
  memcpy(pbuffer, data, length);
  // add null at end to be sure
  *(pbuffer + length) = 0;

  Sections sections;
  Section section;

  std::string string_line;

  const char *delim = "\r\n";
  const char *one_line = std::strtok(pbuffer, delim);
  while (one_line != nullptr)
  {
    if (*one_line == '[')
    {
      if (!section.name.empty())
      {
        sections.push_back(section);
      }
      section.name.clear();
      section.items.clear();

      string_line = one_line + 1;
      auto pos = string_line.find(']');
      assert(pos != std::string::npos);
      if (pos != std::string::npos)
      {
        section.name = string_line.substr(0, pos);
      }
    }
    else if (*one_line == '#' || *one_line == ';')
    {
      // Comment line, do nothing
    }
    else if (*one_line) // string legnth is not 0
    {
      if (section.name.empty())
        throw std::runtime_error("Invalid INI file");

      string_line = one_line;
      auto pos = string_line.find('=');
      assert(pos != std::string::npos);
      if (pos != std::string::npos)
      {
        auto key = string_line.substr(0, pos);
        auto val = string_line.substr(pos + 1);
        section.items.emplace(key, val);
      }
    }

    one_line = std::strtok(nullptr, delim);
  }
  if (!section.name.empty())
  {
    sections.push_back(section);
  }

  return sections;
}

Sections read_ini(const std::string &path)
{
  foder::FileLoader file_loader{path};
  // load will throw if error while opening
  auto ini_data = file_loader.load();

  return read_ini(ini_data.data(), ini_data.size());
}

} // namespace crew
