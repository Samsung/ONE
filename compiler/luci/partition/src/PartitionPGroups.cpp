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

#include "PartitionPGroups.h"
#include "PartitionIR.h"
#include "CircleOpCode.h"

#include "luci/Partition.h"
#include "luci/Log.h"
#include "luci/LogHelper.h"

#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleNodeVisitor.h>

#include <loco.h>

namespace
{

class IsVirtualNode final : public luci::CircleNodeVisitor<bool>
{
public:
  bool visit(const luci::CircleInput *) final { return true; }
  bool visit(const luci::CircleOutput *) final { return true; }
  // For multiple outputs
  bool visit(const luci::CircleCustomOut *) final { return true; }
  bool visit(const luci::CircleIfOut *) final { return true; }
  bool visit(const luci::CircleNonMaxSuppressionV4Out *) final { return true; }
  bool visit(const luci::CircleNonMaxSuppressionV5Out *) final { return true; }
  bool visit(const luci::CircleSplitOut *) final { return true; }
  bool visit(const luci::CircleSplitVOut *) final { return true; }
  bool visit(const luci::CircleTopKV2Out *) final { return true; }
  bool visit(const luci::CircleUniqueOut *) final { return true; }
  bool visit(const luci::CircleUnpackOut *) final { return true; }
  bool visit(const luci::CircleWhileOut *) final { return true; }
  // For inputs not used
  bool visit(const luci::CircleOutputExclude *) final { return true; }
  bool visit(const luci::CircleVariable *) final { return true; }
  // TODO add all virtual nodes

  // default is false
  bool visit(const luci::CircleNode *) final { return false; }
};

bool check_allocate_partition(const luci::CircleNode *node)
{
  IsVirtualNode query;
  if (node->accept(&query))
    return false;
  /**
   * @note About CircleConst
   *       CirleConst acts like a part of some CircleNode and managing mulitiple
   *       used(referenced) CircleConst is a bit difficult if it's used across
   *       different PGroup. So we treat this different to other types.
   *       https://github.com/Samsung/ONE/issues/6230#issuecomment-809802813
   */
  if (dynamic_cast<const luci::CircleConst *>(node) != nullptr)
    return false;
  return true;
}

} // namespace

namespace
{

std::string group_from_partition(const luci::CircleNode *node,
                                 const luci::PartitionTable &partition)
{
  LOGGER(l);

  auto group = partition.default_group;

  std::string opcodename; // opcodename or opname

  switch (partition.comply)
  {
    case luci::PartitionTable::COMPLY::OPCODE:
    {
      opcodename = luci::opcode_name(node);
      assert(!opcodename.empty());

      auto it = partition.byopcodes.find(opcodename);
      if (it != partition.byopcodes.end())
        group = it->second;
      break;
    }
    case luci::PartitionTable::COMPLY::OPNAME:
    {
      opcodename = node->name();
      assert(!opcodename.empty());

      auto it = partition.byopnames.find(opcodename);
      if (it != partition.byopnames.end())
        group = it->second;
      break;
    }

    default:
      throw std::runtime_error("Unsupported partition.comply");
  }

  INFO(l) << "Op: " << node->name() << ": " << opcodename << ", " << node << ", " << group
          << std::endl;

  return group;
}

class IsVirtualInputNode final : public luci::CircleNodeVisitor<bool>
{
public:
  // TODO check CircleOutputDummy
  bool visit(const luci::CircleOutputExclude *) final { return true; }
  bool visit(const luci::CircleVariable *) final { return true; }

  // default is false
  bool visit(const luci::CircleNode *) final { return false; }
};

class IsMultiOutputNode final : public luci::CircleNodeVisitor<bool>
{
public:
  bool visit(const luci::CircleCustom *) final { return true; }
  bool visit(const luci::CircleIf *) final { return true; }
  bool visit(const luci::CircleNonMaxSuppressionV4 *) final { return true; }
  bool visit(const luci::CircleNonMaxSuppressionV5 *) final { return true; }
  bool visit(const luci::CircleSplit *) final { return true; }
  bool visit(const luci::CircleSplitV *) final { return true; }
  bool visit(const luci::CircleTopKV2 *) final { return true; }
  bool visit(const luci::CircleUnique *) final { return true; }
  bool visit(const luci::CircleUnpack *) final { return true; }
  bool visit(const luci::CircleWhile *) final { return true; }
  // default is false
  bool visit(const luci::CircleNode *) final { return false; }
};

void append(luci::CircleNode *node, luci::PGroups *pgroups, const std::string &group, uint32_t idx)
{
  auto pgroup = std::make_unique<luci::PGroup>();
  pgroup->group = group;
  pgroup->id = idx + 1;

  auto pnode = std::make_unique<luci::PNode>();
  pnode->node = node;
  pnode->group = group;
  pnode->pgroup = pgroup.get();

  pgroup->pnodes.push_back(std::move(pnode));

  IsVirtualInputNode queryvi;
  // Set input of PGroup
  for (uint32_t in = 0; in < node->arity(); ++in)
  {
    auto input = loco::must_cast<luci::CircleNode *>(node->arg(in));
    if (input->accept(&queryvi))
    {
      auto pnode_in = std::make_unique<luci::PNode>();
      pnode_in->node = input;
      pnode_in->group = group;
      pnode_in->pgroup = pgroup.get();

      pgroup->pnodes.push_back(std::move(pnode_in));

      pgroups->node2group[input] = group;
    }
    else
    {
      // this input maybe CircleInput in source graph
      // --> not confident this is safe
      pgroup->inputs.push_back(input);
    }
  }

  IsMultiOutputNode query;
  if (node->accept(&query))
  {
    // Include CircleXXXOut virtual nodes in this group
    auto succs = loco::succs(node);
    for (auto &succ_node : succs)
    {
      auto nodeout = loco::must_cast<luci::CircleNode *>(succ_node);

      auto pnode_out = std::make_unique<luci::PNode>();
      pnode_out->node = nodeout;
      pnode_out->group = group;
      pnode_out->pgroup = pgroup.get();

      pgroup->pnodes.push_back(std::move(pnode_out));

      pgroups->node2group[nodeout] = group;

      pgroup->outputs.push_back(nodeout);
    }
  }
  else
  {
    // Set output of PGroup: node itself
    pgroup->outputs.push_back(node);
  }

  pgroups->node2group[node] = group;
  pgroups->id2pgroup[pgroup->id] = pgroup.get();

  pgroups->pgroups.push_back(std::move(pgroup));
}

} // namespace

namespace luci
{

std::unique_ptr<luci::PGroups> produce_pgroups(const luci::Module *source,
                                               const luci::PartitionTable &partition)
{
  assert(source != nullptr);
  // NOTE Only main graph (subgraph index 0) will be partitioned.
  // Other subgraphs will follow the owner (IF/WHILE/...) group

  LOGGER(l);

  auto pgroups = std::make_unique<luci::PGroups>();

  pgroups->default_group = partition.default_group;

  // Create a PGroup per CircleNode: each PGroup will have one CircleNode
  auto graph = source->graph();
  auto nodes = graph->nodes();
  for (uint32_t idx = 0; idx < nodes->size(); ++idx)
  {
    auto node = loco::must_cast<luci::CircleNode *>(nodes->at(idx));

    // check if node is normal node that we are interested
    if (check_allocate_partition(node))
    {
      auto group = group_from_partition(node, partition);

      append(node, pgroups.get(), group, idx);
    }
    else
    {
      INFO(l) << "Skip Op: " << node->name() << std::endl;
      // record as default group
      pgroups->node2group[node] = partition.default_group;
    }
  }

  return pgroups;
}

} // namespace luci
