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

#ifndef __NEST_EXPR_NODE_H__
#define __NEST_EXPR_NODE_H__

#include "nest/expr/Macro.h"
#include "nest/expr/Forward.h"
#include "nest/expr/Visitor.h"

#include <stdexcept>

namespace nest
{
namespace expr
{

struct Node
{
  virtual ~Node() = default;

#define EXPR(Tag)                                                                       \
  virtual const NEST_EXPR_CLASS_NAME(Tag) * NEST_EXPR_CAST_METHOD_NAME(Tag)(void) const \
  {                                                                                     \
    return nullptr;                                                                     \
  }
#include "nest/expr/Node.def"
#undef EXPR

  template <typename T> T accept(Visitor<T> *v)
  {
#define EXPR(Tag)                                 \
  if (auto e = NEST_EXPR_CAST_METHOD_NAME(Tag)()) \
  {                                               \
    return v->visit(e);                           \
  }
#include "nest/expr/Node.def"
#undef EXPR

    throw std::runtime_error{"unreachable"};
  }

  template <typename T> T accept(Visitor<T> &v) { return accept(&v); }
};

} // namespace expr
} // namespace nest

#endif // __NEST_EXPR_NODE_H__
