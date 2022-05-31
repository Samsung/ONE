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
#include "crew/PConfigIniDump.h"

#include <foder/FileLoader.h>

#include <cassert>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace crew
{

namespace
{

std::string filter_escape(const std::string &source)
{
  std::string key = source;

  // if key is surrounded with quotation
  // TODO for quotation

  // if key has '\\' + ';', remove '\\'
  auto pos = key.find("\\;");
  while (pos != std::string::npos)
  {
    auto k1 = key.substr(0, pos);
    auto k2 = key.substr(pos + 1);
    key = k1 + k2;
    pos = key.find("\\;");
  }

  return key;
}

} // namespace

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
        key = filter_escape(key);
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

namespace
{

std::string replace(const std::string &source, const char token, const std::string &replace)
{
  std::string key = source;
  auto pos = key.find(token);
  std::vector<std::string> subkeys;
  while (pos != std::string::npos)
  {
    auto sub = key.substr(0, pos);
    subkeys.push_back(sub);
    key = key.substr(pos + 1);
    pos = key.find(token);
  }
  if (!key.empty())
  {
    subkeys.push_back(key);
  }
  std::string key_new;
  for (auto &sub : subkeys)
  {
    if (!key_new.empty())
      key_new = key_new + replace;
    key_new = key_new + sub;
  }
  return key_new;
}

Sections insert_escape(const Sections &inputs)
{
  Sections sections;

  // for all section in sections;
  // if key has ';' then replace with '\;'
  for (auto &input : inputs)
  {
    Section section;
    section.name = input.name;

    for (auto &item : input.items)
    {
      auto key = item.first;
      auto value = item.second;

      auto pos = key.find(';');
      if (pos != std::string::npos)
      {
        auto key_new = replace(key, ';', "\\;");
        section.items[key_new] = value;
      }
      else
      {
        section.items[key] = value;
      }
    }
    sections.push_back(section);
  }

  return sections;
}

} // namespace

void write_ini(std::ostream &os, const Sections &sections)
{
  std::stringstream ss;

  auto processed = insert_escape(sections);

  ss << processed;

  std::string strss = ss.str();

  os.write(strss.c_str(), strss.length());
}

void write_ini(const std::string &filepath, const Sections &sections)
{
  std::ofstream fs(filepath.c_str(), std::ofstream::binary | std::ofstream::trunc);
  if (not fs.good())
  {
    std::string msg = "Failed to create file: " + filepath;
    throw std::runtime_error(msg);
  }

  write_ini(fs, sections);

  fs.close();
}

Section find(const Sections &sections, const std::string &name)
{
  for (auto &section : sections)
  {
    if (section.name == name)
      return section;
  }
  Section not_found;
  return not_found;
}

std::string find(const Section &section, const std::string &key)
{
  for (auto &item : section.items)
  {
    if (item.first == key)
      return item.second;
  }
  return "";
}

} // namespace crew
