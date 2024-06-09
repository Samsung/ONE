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

#ifndef __CREW_PCONFIG_JSON_H__
#define __CREW_PCONFIG_JSON_H__

#include <ostream>
#include <string>
#include <vector>
#include <cstdint>

namespace crew
{

class JsonExport
{
public:
  JsonExport(std::ostream &os) : _os(os) {}

private:
  void indent(void);

public:
  void open_brace(void);
  void open_brace(const std::string &key);
  void open_bracket(const std::string &key);
  void close_bracket(bool cont);
  void close_brace(bool cont);
  void key_val(const std::string &key, const std::string &value, bool cont);
  void key_val(const std::string &key, const std::vector<std::string> &l, bool cont);

private:
  std::ostream &_os;
  uint32_t _indent = 0;
};

} // namespace crew

#endif // __CREW_PCONFIG_JSON_H__
