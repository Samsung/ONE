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

#ifndef __NEXT_EXPR_DEREF_NODE_H__
#define __NEXT_EXPR_DEREF_NODE_H__

#include "nest/DomainID.h"

#include "nest/expr/Subscript.h"

namespace nest
{
namespace expr
{

class DerefNode final : public Node
{
public:
  template <typename... Args>
  DerefNode(const DomainID &id, Args &&...indicies) : _id{id}, _sub{std::forward<Args>(indicies)...}
  {
    // DO NOTHING
  }

public:
  const DerefNode *asDeref(void) const override { return this; }

public:
  const DomainID &id(void) const { return _id; }
  const Subscript &sub(void) const { return _sub; }

private:
  DomainID const _id;
  Subscript const _sub;
};

} // namespace expr
} // namespace nest

#endif // __NEXT_EXPR_DEREF_NODE_H__
