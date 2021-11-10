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

class FindGroupToFollow final : public luci::CircleNodeVisitor<const std::string &>
{
public:
  FindGroupToFollow(const luci::PartitionTable &partition, luci::PGroups *pgroups)
    : _partition(partition), _pgroups(pgroups)
  {
    // NOTHING TODO
  }

private:
  const std::string &groupof(const luci::CircleNode *input) const
  {
    auto group = _pgroups->node2group[input];
    assert(not group.empty());
    if (group.empty())
      return _partition.default_group;
    return _pgroups->node2group[input];
  }

public:
#define IMPLEMENT(CLASS)                                             \
  const std::string &visit(const luci::CLASS *node) final            \
  {                                                                  \
    auto input = loco::must_cast<luci::CircleNode *>(node->input()); \
    return groupof(input);                                           \
  }

  IMPLEMENT(CircleCustomOut);
  IMPLEMENT(CircleIfOut);
  IMPLEMENT(CircleNonMaxSuppressionV4Out);
  IMPLEMENT(CircleNonMaxSuppressionV5Out);
  IMPLEMENT(CircleSplitOut);
  IMPLEMENT(CircleSplitVOut);
  IMPLEMENT(CircleTopKV2Out);
  IMPLEMENT(CircleUniqueOut);
  IMPLEMENT(CircleUnpackOut);
  IMPLEMENT(CircleWhileOut);

#undef IMPLEMENT

  // return empty for nothing to do
  const std::string &visit(const luci::CircleNode *) final { return _empty_str; }

private:
  const luci::PartitionTable &_partition;
  luci::PGroups *_pgroups = nullptr;
  std::string _empty_str;
};

} // namespace

namespace
{

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

  // Set input of PGroup
  for (uint32_t in = 0; in < node->arity(); ++in)
  {
    auto input = loco::must_cast<luci::CircleNode *>(node->arg(in));
    // this input maybe CircleInput in source graph
    // --> not confident this is safe
    pgroup->inputs.push_back(input);
  }
  // Set output of PGroup: node itself or multiple virtual outputs
  // TODO support multiple virtual outputs
  pgroup->outputs.push_back(node);

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

      append(node, pgroups.get(), group, idx);
#if 0
      auto pgroup = std::make_unique<luci::PGroup>();
      pgroup->group = group;
      pgroup->id = idx + 1;

      auto pnode = std::make_unique<luci::PNode>();
      pnode->node = node;
      pnode->group = group;
      pnode->pgroup = pgroup.get();

      pgroup->pnodes.push_back(std::move(pnode));

      // Set input of PGroup
      for (uint32_t in = 0; in < node->arity(); ++in)
      {
        auto input = loco::must_cast<luci::CircleNode *>(node->arg(in));
        // this input maybe CircleInput in source graph
        // --> not confident this is safe
        pgroup->inputs.push_back(input);
      }
      // Set output of PGroup: node itself or multiple virtual outputs
      // TODO support multiple virtual outputs
      pgroup->outputs.push_back(node);

      pgroups->node2group[node] = group;
      pgroups->id2pgroup[pgroup->id] = pgroup.get();

      pgroups->pgroups.push_back(std::move(pgroup));
#endif
    }
    else
    {
      INFO(l) << "Skip Op: " << node->name() << std::endl;
      // record as default group
      pgroups->node2group[node] = partition.default_group;
    }
  }

  // handle for virtual nodes like multiple outputs
  // these nodes should follow group of the input
  for (uint32_t idx = 0; idx < nodes->size(); ++idx)
  {
    auto node = loco::must_cast<luci::CircleNode *>(nodes->at(idx));

    // for virtual nodes like CircleUnpackOut should follow it's input (owner)
    // or just set to default
    FindGroupToFollow query(partition, pgroups.get());
    const auto &group = node->accept(&query);
    if (not group.empty())
    {
      append(node, pgroups.get(), group, idx);
    }
  }

  return std::move(pgroups);
}

} // namespace luci
