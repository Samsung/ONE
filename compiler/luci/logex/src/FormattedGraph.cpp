/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "CircleNodeSummaryBuilder.h"
#include "luci/FormattedGraph.h"

#include <luci/IR/CircleDialect.h>
#include <luci/IR/CircleNodes.h>

#include <pepper/str.h>

#include <cassert>
#include <sstream>
#include <vector>

using namespace luci;

namespace
{

std::string circle_opname(uint32_t opnum)
{
  static const std::string prefix{"circle."};

  switch (static_cast<luci::CircleOpcode>(opnum))
  {
#define CIRCLE_NODE(OPCODE, CLASS) \
  case luci::CircleOpcode::OPCODE: \
    return prefix + #OPCODE;
#define CIRCLE_VNODE CIRCLE_NODE
#include <luci/IR/CircleNodes.lst>
#undef CIRCLE_VNODE
#undef CIRCLE_NODE
    default:
      break;
  };

  return prefix + "Invalid";
}

// CircleNodeSummaryBuilder with default implementation
class CircleNodeSummaryBuilderBase : public locop::NodeSummaryBuilder
{
public:
  CircleNodeSummaryBuilderBase(const locop::SymbolTable *tbl) : _tbl{tbl}
  {
    // DO NOTHING
  }

public:
  bool build(const loco::Node *, locop::NodeSummary &s) const final;

protected:
#define CIRCLE_NODE(OPCODE, CLASS) \
  virtual bool summary(const CLASS *, locop::NodeSummary &) const { return false; }
#define CIRCLE_VNODE CIRCLE_NODE
#include <luci/IR/CircleNodes.lst>
#undef CIRCLE_VNODE
#undef CIRCLE_NODE

protected:
  const locop::SymbolTable *tbl(void) const { return _tbl; }

private:
  const locop::SymbolTable *_tbl;
};

#undef IMPLEMENT

bool CircleNodeSummaryBuilderBase::build(const loco::Node *node, locop::NodeSummary &s) const
{
  if (node->dialect() != luci::CircleDialect::get())
    return false;

  auto ptr_to_str = [](const void *ptr) {
    std::stringstream ss;
    ss << ptr;
    return ss.str();
  };

  auto add_comment = [&]() {
    auto cnode = loco::must_cast<const luci::CircleNode *>(node);
    s.opname(circle_opname(node->opnum()));
    s.comments().append("[" + cnode->name() + "] = " + ptr_to_str(node));
  };

#define CIRCLE_NODE(OPCODE, CLASS)                     \
  if (dynamic_cast<const CLASS *>(node))               \
  {                                                    \
    if (summary(dynamic_cast<const CLASS *>(node), s)) \
    {                                                  \
      add_comment();                                   \
      return true;                                     \
    }                                                  \
  }
#define CIRCLE_VNODE CIRCLE_NODE
#include <luci/IR/CircleNodes.lst>
#undef CIRCLE_VNODE
#undef CIRCLE_NODE

  return false;
}

} // namespace

namespace luci
{

bool NodeSummaryBuilder::build(const loco::Node *node, locop::NodeSummary &s) const
{
  if (locop::CanonicalNodeSummaryBuilder(_tbl).build(node, s))
  {
    return true;
  }

  if (CircleNodeSummaryBuilder().build(node, _tbl, s))
  {
    return true;
  }

  return false;
}

} // namespace luci
