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

#include "luci/RawGraphDumper.h"

#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleNodeVisitor.h>

#include <ostream>

namespace
{

struct RawGraphDumper final : public luci::CircleNodeVisitor<void>
{
  RawGraphDumper(std::ostream &os) : _os(os) {}

#define CIRCLE_NODE(CIRCLE_CLASS)                                               \
  void visit(const luci::CIRCLE_CLASS *node) final                              \
  {                                                                             \
    _os << #CIRCLE_CLASS << " " << node->name() << " at " << node << std::endl; \
  }

  CIRCLE_NODE(CircleAdd);
  CIRCLE_NODE(CircleInput);
  CIRCLE_NODE(CircleOutput);
  CIRCLE_NODE(CircleOutputDummy);

#undef CIRCLE_NODE

  void visit(const luci::CircleNode *node) final
  {
    _os << "CircleNode "
        << " " << node->name() << " at " << node << std::endl;
  }

  std::ostream &_os;
};

void dump_raw_node(std::ostream &os, luci::CircleNode *node)
{
  RawGraphDumper d(os);
  node->accept(&d);
}

} // namespace

std::ostream &operator<<(std::ostream &os, luci::RawDumpGraph r)
{
  auto nodes = r._g->nodes();
  for (uint32_t idx = 0; idx < nodes->size(); ++idx)
  {
    auto node = nodes->at(idx);
    auto circle_node = dynamic_cast<luci::CircleNode *>(node);
    assert(circle_node != nullptr);
    dump_raw_node(os, circle_node);
  }

  return os;
}

std::ostream &operator<<(std::ostream &os, luci::RawDumpNode r)
{
  dump_raw_node(os, r._n);

  return os;
}
