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

#include "hermes/core/Message.h"

#include <cassert>

namespace hermes
{

MessageText::MessageText(std::stringstream &ss)
{
  while (!ss.eof())
  {
    assert(ss.good());

    std::string line;
    std::getline(ss, line);

    // Trim the last empty line (by std::endl)
    if (ss.eof() && line.empty())
    {
      break;
    }

    _lines.emplace_back(line);
  }
}

} // namespace hermes
