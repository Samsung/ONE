/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __MOCO_FRONTEND_ONNX_GRAPHBUILDERCONTEXT_H__
#define __MOCO_FRONTEND_ONNX_GRAPHBUILDERCONTEXT_H__

#include <loco.h>

#include <map>
#include <string>
#include <vector>

namespace moco
{
namespace onnx
{

/**
 * @brief Class to store relations of Nodes and string names
 */
class SymbolTable
{
public:
  /**
   * @brief  Registers one node for a name
   */
  void enroll(const std::string &node_name, loco::Node *node);
  /**
   * @brief  Queries enrolled(registered) with name and return node if found
   *         Will throw runtime_error if not found
   *         Table is independent with registering with list()
   */
  loco::Node *node(const std::string &node_name);

  /**
   * @brief  Registers multiple (appends) names for a node
   *         Table is independent with registering with enroll()
   */
  void list(loco::Node *node, const std::string &name);
  /**
   * @brief  Returns number of listed(registered) names for a node
   */
  unsigned size(loco::Node *node);
  /**
   * @brief  Queries listed(registered) with node and index(from 0 to size-1)
   *         Will throw runtime_error if node is not found or index is out of bounds
   */
  const std::string &name(loco::Node *node, unsigned index);

private:
  using MapNameNode_t = std::map<std::string, loco::Node *>;
  using MapNodeNames_t = std::map<loco::Node *, std::vector<std::string>>;

  MapNameNode_t _namenode;
  MapNodeNames_t _nodenames;
};

/**
 * @brief Class to store context to build IR from onnx
 */
class GraphBuilderContext
{
public:
  GraphBuilderContext(loco::Graph *g, SymbolTable *nodes, SymbolTable *input_names)
    : _g(g), _nodes(nodes), _input_names(input_names)
  {
    // DO NOTHING
  }

  GraphBuilderContext(const GraphBuilderContext &) = delete;
  GraphBuilderContext(GraphBuilderContext &&) = delete;

public:
  loco::Graph *graph() { return _g; }
  SymbolTable *nodes() { return _nodes; }
  SymbolTable *input_names() { return _input_names; }

private:
  loco::Graph *_g;
  SymbolTable *_nodes;
  SymbolTable *_input_names;
};

} // namespace onnx
} // namespace moco

#endif // __MOCO_FRONTEND_ONNX_GRAPHBUILDERCONTEXT_H__
