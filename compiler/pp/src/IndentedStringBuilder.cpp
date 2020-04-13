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

#include "pp/IndentedStringBuilder.h"

#include <algorithm>
#include <cassert>

namespace pp
{

void IndentedStringBuilder::increase(void)
{
  // TODO Check overflow
  ++_level;
}

void IndentedStringBuilder::decrease(void)
{
  assert(_level > 0);
  --_level;
}

std::string IndentedStringBuilder::build(const std::string &content)
{
  assert(std::find(content.begin(), content.end(), '\n') == content.end());

  const char c = ' ';
  const size_t space_per_indent_level = 2;
  const size_t space_count = space_per_indent_level * _level;

  return std::string(space_count, c) + content;
}

} // namespace pp
