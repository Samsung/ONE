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

#ifndef __MOCO_IMPORT_GRAPH_BUILDER_CONTEXT_H__
#define __MOCO_IMPORT_GRAPH_BUILDER_CONTEXT_H__

#include <moco/Names.h>

#include <loco.h>

#include <tensorflow/core/framework/graph.pb.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace moco
{

/**
 * @brief Class to store and query tensorflow::NodeDef* with string name key
 */
class NodeDefTable
{
public:
  /**
   * @brief Registers a name with corresponding tensorflow::NodeDef*
   */
  void enroll(const std::string &node_name, const tensorflow::NodeDef *node);
  /**
   * @brief Queries enrolled(registered) with name and return node if found
   *        Will throw runtime_error if not found
   */
  const tensorflow::NodeDef *node(const std::string &node_name) const;

private:
  using MapNameNode_t = std::map<std::string, const tensorflow::NodeDef *>;

  MapNameNode_t _table;
};

/**
 * @brief Class to store and query loco::Node* with string name key
 */
class SymbolTable
{
public:
  /**
   * @brief  Registers a name with corresponding loco::Node *
   */
  void enroll(const TensorName &tensor_name, loco::Node *node);
  /**
   * @brief  Queries enrolled(registered) with name and return node if found
   *         Will throw runtime_error if not found
   */
  loco::Node *node(const TensorName &tensor_name) const;

private:
  using MapNameNode_t = std::map<TensorName, loco::Node *, TensorNameCompare>;

  MapNameNode_t _table;
};

/**
 * @brief Interface to connect the graph
 */
class GraphUpdate
{
public:
  virtual ~GraphUpdate() = default;

public:
  /**
   * @brief  Do the graph input connections using the SymbolTable
   */
  virtual void input(const SymbolTable *) const = 0;
};

/**
 * @brief Class to store GraphUpdate objects
 */
class UpdateQueue final
{
public:
  /**
   * @brief  Registers GraphUpdate objects
   */
  void enroll(std::unique_ptr<GraphUpdate> &&update);

public:
  using Queue = std::vector<std::unique_ptr<GraphUpdate>>;

  const Queue &queue() const { return _queue; }

private:
  Queue _queue;
};

/**
 * @brief Class to store context to build loco graph IR from TensorFlow
 */
class GraphBuilderContext
{
public:
  GraphBuilderContext(loco::Graph *g, NodeDefTable *nodedef, SymbolTable *tensor_names,
                      UpdateQueue *updates)
    : _g(g), _nodedef(nodedef), _tensor_names(tensor_names), _updates(updates)
  {
    // DO NOTHING
  }

  GraphBuilderContext(const GraphBuilderContext &) = delete;
  GraphBuilderContext(GraphBuilderContext &&) = delete;

public:
  loco::Graph *graph() { return _g; }
  NodeDefTable *nodedef() { return _nodedef; }
  SymbolTable *tensor_names() { return _tensor_names; }
  UpdateQueue *updates() { return _updates; }

private:
  loco::Graph *_g;
  NodeDefTable *_nodedef;
  SymbolTable *_tensor_names;
  UpdateQueue *_updates;
};

} // namespace moco

#endif // __MOCO_IMPORT_GRAPH_BUILDER_CONTEXT_H__
