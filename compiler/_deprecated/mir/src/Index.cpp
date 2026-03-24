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

#include "mir/Index.h"

#include <algorithm>

namespace mir
{

Index &Index::resize(int32_t size)
{
  _indices.resize(size);
  return *this;
}

Index &Index::fill(int32_t index)
{
  std::fill(std::begin(_indices), std::end(_indices), index);
  return (*this);
}

std::ostream &operator<<(std::ostream &s, const Index &idx)
{
  s << "[ ";
  for (int32_t i = 0; i < idx.rank(); ++i)
  {
    if (i != 0)
      s << ", ";
    s << idx.at(i);
  }
  s << "]";

  return s;
}

} // namespace mir
