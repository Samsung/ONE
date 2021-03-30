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
  BuiltinOperator visit(const CircleAdd *) final { return BuiltinOperator_ADD; }
  BuiltinOperator visit(const CircleDiv *) final { return BuiltinOperator_DIV; }
  BuiltinOperator visit(const CircleMean *) final { return BuiltinOperator_MEAN; }
  BuiltinOperator visit(const CircleMul *) final { return BuiltinOperator_MUL; }
  BuiltinOperator visit(const CirclePow *) final { return BuiltinOperator_POW; }
  BuiltinOperator visit(const CircleRsqrt *) final { return BuiltinOperator_RSQRT; }
  BuiltinOperator visit(const CircleSqrt *) final { return BuiltinOperator_SQRT; }
  BuiltinOperator visit(const CircleSquaredDifference *) final
  {
    return BuiltinOperator_SQUARED_DIFFERENCE;
  }
  BuiltinOperator visit(const CircleSub *) final { return BuiltinOperator_SUB; }

  // NOTE only builtin operators should be called (NOT virtual nodes)
};

using PCSTR = const char *;

class QueryCircleName final : public luci::CircleNodeVisitor<PCSTR>
{
public:
  PCSTR visit(const luci::CircleConst *) final { return "CIRCLE_CONST"; }

  // default is null
  PCSTR visit(const luci::CircleNode *) final { return nullptr; }
};

} // namespace

namespace luci
{

std::string opcode_name(const CircleNode *node)
{
  QueryCircleName qcn;
  auto cname = node->accept(&qcn);
  if (cname != nullptr)
    return std::string(cname);

  QueryOpCode qoc;
  auto opcode = node->accept(&qoc);
  auto name = circle::EnumNameBuiltinOperator(opcode);
  return std::string(name);
}

} // namespace luci
