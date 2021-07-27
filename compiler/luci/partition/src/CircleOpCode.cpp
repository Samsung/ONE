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
// NOTE only circle operator may have BuiltinOperator_XXX
#define CIRCLE_NODE(OPCODE, CIRCLE_CLASS) \
  BuiltinOperator visit(const luci::CIRCLE_CLASS *) final { return BuiltinOperator_##OPCODE; }
#define CIRCLE_VNODE(OPCODE, CIRCLE_CLASS)

#include "luci/IR/CircleNodes.lst"
#undef CIRCLE_VNODE
#undef CIRCLE_NODE

  // NOTE only builtin operators should be called (NOT virtual nodes)
};

class QueryCircleName final : public luci::CircleNodeVisitor<const char *>
{
public:
// NOTE provide names for circle virtual nodes
#define CIRCLE_NODE(OPCODE, CIRCLE_CLASS)
#define CIRCLE_VNODE(OPCODE, CIRCLE_CLASS) \
  const char *visit(const luci::CIRCLE_CLASS *) final { return #OPCODE; }

#include "luci/IR/CircleNodes.lst"
#undef CIRCLE_VNODE
#undef CIRCLE_NODE

  // default is null
  const char *visit(const luci::CircleNode *) final { return nullptr; }
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
