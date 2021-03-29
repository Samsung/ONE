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

#include "PConfigJson.h"

#include <iostream>
#include <string>
#include <vector>

namespace
{

const char _CLF = '\n'; // Control Line Feed
const char _DQU = '\"'; // Double QUotation

} // namespace

namespace crew
{

void JsonExport::indent(void)
{
  for (uint32_t i = 0; i < _indent; ++i)
    _os << "  ";
}

void JsonExport::open_brace(void)
{
  indent();

  _os << "{" << _CLF;
  _indent++;
}

void JsonExport::open_brace(const std::string &key)
{
  indent();

  _os << _DQU << key << _DQU << " : {" << _CLF;
  _indent++;
}

void JsonExport::open_bracket(const std::string &key)
{
  indent();

  _os << _DQU << key << _DQU << " : [" << _CLF;
  _indent++;
}

void JsonExport::close_bracket(bool cont)
{
  _indent--;
  indent();

  _os << "]";
  if (cont)
    _os << ",";
  _os << _CLF;
}

void JsonExport::close_brace(bool cont)
{
  _indent--;
  indent();

  _os << "}";
  if (cont)
    _os << ",";
  _os << _CLF;
}

void JsonExport::key_val(const std::string &key, const std::string &value, bool cont)
{
  indent();

  _os << _DQU << key << _DQU << " : " << _DQU << value << _DQU;
  if (cont)
    _os << ",";
  _os << _CLF;
}

void JsonExport::key_val(const std::string &key, const std::vector<std::string> &l, bool cont)
{
  indent();

  _os << _DQU << key << _DQU << " : [ ";
  bool comma = false;
  for (auto &v : l)
  {
    if (comma)
      _os << ", ";
    else
      comma = true;
    _os << _DQU << v << _DQU;
  }
  _os << " ]";
  if (cont)
    _os << ",";
  _os << _CLF;
}

} // namespace crew
