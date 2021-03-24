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

#include "PartitionIRDump.h"

#include "CircleOpCode.h"

#include <iostream>

namespace luci
{

void dump(std::ostream &os, const PNode *pnode)
{
  os << "PNode: " << pnode->group << ", " << pnode->node << ":" << luci::opcode_name(pnode->node)
     << ":" << pnode->node->name() << std::endl;
}

void dump(std::ostream &os, const PGraph *pgraph)
{
  os << "--- PGraph: " << pgraph->group << std::endl;
  os << "Input(s): ";
  for (auto &node_in : pgraph->inputs)
    os << node_in->name() << " ";
  os << std::endl;
  for (auto &pnode : pgraph->pnodes)
  {
    dump(os, pnode.get());
  }
  os << "Output(s): ";
  for (auto &node_out : pgraph->outputs)
    os << node_out->name() << " ";
  os << std::endl;
}

void dump(std::ostream &os, const PGraphs *pgraphs)
{
  for (auto &pgraph : pgraphs->pgraphs)
  {
    dump(os, pgraph.get());
  }
  os << "--- Node2Group items: " << std::endl;
  for (auto it = pgraphs->node2group.begin(); it != pgraphs->node2group.end(); ++it)
  {
    auto node = it->first;
    auto group = it->second;
    os << "  Node: " << node << "(" << node->name() << "): " << group << std::endl;
  }
}

} // namespace luci

std::ostream &operator<<(std::ostream &os, const luci::PGraphs *pgraphs)
{
  luci::dump(os, pgraphs);
  return os;
}
