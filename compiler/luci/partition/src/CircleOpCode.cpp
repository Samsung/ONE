/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "CircleOpCode.h"

#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleNodeVisitor.h>

#include <mio/circle/schema_generated.h>

namespace
{

using namespace luci;
using namespace circle;

class QueryOpCode final : public CircleNodeVisitor<BuiltinOperator>
{
public:
  // TODO add CircleNodes
  BuiltinOperator visit(const CircleRsqrt *) final { return BuiltinOperator_RSQRT; }
  BuiltinOperator visit(const CircleSqrt *) final { return BuiltinOperator_SQRT; }

  // NOTE only builtin operators should be called (NOT virtual nodes)
};

} // namespace

namespace luci
{

std::string opcode_name(const CircleNode *node)
{
  QueryOpCode qoc;
  auto opcode = node->accept(&qoc);
  auto name = circle::EnumNameBuiltinOperator(opcode);
  return std::string(name);
}

} // namespace luci
