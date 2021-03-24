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

#ifndef __LUCI_PARTITION_IR_H__
#define __LUCI_PARTITION_IR_H__

#include <luci/IR/CircleNodes.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace luci
{

struct PGraph;

/**
 * @brief Partition Node with CircleNode with group name
 * @note  node just points to source luci::CircleNode, NOT the cloned node
 *        CloneContext is used to find cloned node from source node
 */
struct PNode
{
  const luci::CircleNode *node = nullptr;
  std::string group;

  const PGraph *pgraph = nullptr;
};

/**
 * @brief Partition Graph with Partition Nodes of same group and graph I/Os
 */
struct PGraph
{
  std::vector<std::unique_ptr<PNode>> pnodes;
  std::string group;
  uint32_t id = 0;

  // I/O while partitioning
  std::vector<luci::CircleNode *> inputs;
  std::vector<luci::CircleNode *> outputs;
};

struct PGraphs
{
  std::vector<std::unique_ptr<PGraph>> pgraphs;

  // node2group is to find group key from source node
  std::map<const luci::CircleNode *, std::string> node2group;

  // id2pgraph is to find *pgraph from pgraph id
  std::map<uint32_t, PGraph *> id2pgraph;

public:
  /**
   * @brief return a copy of PGraphs
   */
  std::unique_ptr<PGraphs> make_copy(void) const;

  /**
   * @brief return group key of node, empty string if not found
   */
  std::string group_of(luci::CircleNode *node) const;

  /**
   * @brief return holding pgraph of node, nullptr if not found
   */
  const PGraph *pgraph_of(luci::CircleNode *node) const;
};

} // namespace luci

#endif // __LUCI_PARTITION_IR_H__
