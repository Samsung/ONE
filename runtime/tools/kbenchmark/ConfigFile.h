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

#ifndef __KBENCHMARK_CONFIG_FILE_H__
#define __KBENCHMARK_CONFIG_FILE_H__

#include <fstream>
#include <string>
#include <algorithm>
#include <regex>
#include <map>

namespace
{

std::string getName(const std::string &src)
{
  std::string name{src.substr(0, src.find_last_of("."))};

  std::string op;
  size_t pos = 0;
  std::string token{"Main_model_"};
  if ((pos = name.find(token)) != std::string::npos)
  {
    op = name.substr(pos + token.size());
  }
  else if ((pos = name.find("Model_#")) != std::string::npos)
  {
    op = std::regex_replace(name, std::regex("[^0-9]*([0-9]+)_*"), "$2");
  }
  return op;
}

std::string trim(const std::string &src, const std::string &delims)
{
  std::string str{src};
  for (int i = 0; i < delims.size(); ++i)
  {
    str.erase(std::remove(str.begin(), str.end(), delims[i]), str.end());
  }
  return str;
}

} // namespace

namespace kbenchmark
{

using OperationInfo = std::map<std::string, std::string>;

class ConfigFile
{
public:
  using iterator = std::map<int, OperationInfo>::iterator;
  using const_iterator = std::map<int, OperationInfo>::const_iterator;

public:
  ConfigFile(const std::string &config) : _name{getName(config)}
  {
    std::ifstream file(config.c_str());

    std::string line;
    int id;
    std::string key;
    std::string value;
    size_t pos;

    while (std::getline(file, line))
    {
      if (!line.length())
        continue;
      if (line[0] == '#')
        continue;
      if (line[0] == '[')
      {
        id = std::stoi(line.substr(1, line.find(']') - 1));
        continue;
      }
      pos = line.find(':');
      key = line.substr(0, pos);
      value = trim(line.substr(pos + 1), " []");
      _contents[id][key] = value;
    }
  }

  const std::string name(void) { return _name; }

  iterator begin(void) { return _contents.begin(); }
  iterator end(void) { return _contents.end(); }
  const_iterator begin(void) const { return _contents.begin(); }
  const_iterator end(void) const { return _contents.end(); }
  const_iterator cbegin(void) const { return _contents.cbegin(); }
  const_iterator cend(void) const { return _contents.cend(); }

private:
  std::string _name;
  std::map<int, OperationInfo> _contents;
};

} // namespace kbenchmark

#endif // __KBENCHMARK_CONFIG_FILE_H__
