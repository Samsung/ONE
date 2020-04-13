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

#include "pp/LinearDocument.h"

#include <stdexcept>

namespace pp
{

void LinearDocument::indent(void) { _indent.increase(); }
void LinearDocument::unindent(void) { _indent.decrease(); }

void LinearDocument::append(void)
{
  // NOTE Do NOT indent empty lines
  _lines.emplace_back("");
}

void LinearDocument::append(const std::string &line)
{
  if (line.empty())
  {
    append();
    return;
  }

  // Append indentation space(s), and insert the update string to lines
  _lines.emplace_back(_indent.build(line));
}

void LinearDocument::append(const LinearDocument &doc)
{
  for (uint32_t n = 0; n < doc.lines(); ++n)
  {
    // NOTE Do NOT update _lines here and use append method
    append(doc.line(n));
  }
}

const std::string &LinearDocument::line(uint32_t n) const
{
  switch (_direction)
  {
    case Direction::Forward:
    {
      return _lines.at(n);
    }
    case Direction::Reverse:
    {
      return _lines.at(lines() - n - 1);
    }
  }

  throw std::runtime_error{"unreachable"};
}

} // namespace pp
