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

#ifndef __PP_LINEAR_DOCUMENT_H__
#define __PP_LINEAR_DOCUMENT_H__

#include "pp/Format.h"
#include "pp/IndentedStringBuilder.h"
#include "pp/MultiLineText.h"

#include <vector>
#include <string>

#include <type_traits>

namespace pp
{

class LinearDocument final : public MultiLineText
{
public:
  enum class Direction
  {
    Forward,
    Reverse
  };

public:
  LinearDocument() : _direction{Direction::Forward}
  {
    // DO NOTHING
  }

public:
  LinearDocument(const Direction &direction) : _direction{direction}
  {
    // DO NOTHING
  }

public:
  void indent(void);
  void unindent(void);

public:
  void append(void);

public:
  void append(const std::string &line);

  template <typename Derived>
  typename std::enable_if<std::is_base_of<MultiLineText, Derived>::value>::type
  append(const Derived &txt)
  {
    for (uint32_t n = 0; n < txt.lines(); ++n)
    {
      append(txt.line(n));
    }
  }

  template <typename... Args> void append(const Args &...args) { append(fmt(args...)); }

public:
  void append(const LinearDocument &doc);

public:
  uint32_t lines(void) const override { return _lines.size(); }

public:
  const std::string &line(uint32_t n) const override;

private:
  Direction const _direction;
  IndentedStringBuilder _indent;
  std::vector<std::string> _lines;
};

} // namespace pp

#endif // __PP_LINEAR_DOCUMENT_H__
