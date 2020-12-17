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

#ifndef __NEST_EXPR_MUL_NODE_H__
#define __NEST_EXPR_MUL_NODE_H__

#include "nest/expr/Node.h"

#include <memory>

namespace nest
{
namespace expr
{

class MulNode final : public Node
{
public:
  MulNode(const std::shared_ptr<expr::Node> &lhs, const std::shared_ptr<expr::Node> &rhs)
    : _lhs{lhs}, _rhs{rhs}
  {
    // DO NOTHING
  }

public:
  const MulNode *asMul(void) const override { return this; }

public:
  const std::shared_ptr<expr::Node> &lhs(void) const { return _lhs; }
  const std::shared_ptr<expr::Node> &rhs(void) const { return _rhs; }

private:
  std::shared_ptr<expr::Node> const _lhs;
  std::shared_ptr<expr::Node> const _rhs;
};

} // namespace expr
} // namespace nest

#endif // __NEST_EXPR_MUL_NODE_H__
