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

#ifndef __NEST_STMT_PUSH_NODE_H__
#define __NEST_STMT_PUSH_NODE_H__

#include "nest/stmt/Node.h"
#include "nest/Expr.h"

namespace nest
{
namespace stmt
{

class PushNode final : public Node
{
public:
  PushNode(const Expr &expr) : _expr{expr}
  {
    // DO NOTHING
  }

public:
  const PushNode *asPush(void) const override { return this; }

public:
  const Expr &expr(void) const { return _expr; }

private:
  Expr const _expr;
};

} // namespace stmt
} // namespace nest

#endif // __NEST_STMT_PUSH_NODE_H__
