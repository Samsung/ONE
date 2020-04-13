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

#ifndef __NEST_SUBSCRIPT_H__
#define __NEST_SUBSCRIPT_H__

#include "nest/expr/Node.h"

#include <vector>
#include <initializer_list>

#include <memory>

namespace nest
{
namespace expr
{

class Subscript
{
public:
  Subscript(std::initializer_list<std::shared_ptr<Node>> indices) : _indices{indices}
  {
    // DO NOTHING
  }

public:
  uint32_t rank(void) const { return _indices.size(); }

public:
  const std::shared_ptr<expr::Node> &at(uint32_t n) const { return _indices.at(n); }

private:
  std::vector<std::shared_ptr<expr::Node>> _indices;
};

} // namespace expr
} // namespace nest

#endif // __NEST_SUBSCRIPT_H__
