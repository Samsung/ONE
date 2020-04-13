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

#ifndef __LUCI_IMPORT_GRAPH_BUILDER_CONTEXT_H__
#define __LUCI_IMPORT_GRAPH_BUILDER_CONTEXT_H__

#include "CircleReader.h"

#include <luci/IR/CircleNode.h>

#include <loco.h>

#include <map>

namespace luci
{

/*
 * @brief  CircleNode to circle::Operator
 *         To find circle::Operator from CircleNode
 */
class NodeOpFinder
{
public:
  void enroll(CircleNode *node, const circle::Operator *op);

  const circle::Operator *op(CircleNode *node) const;

private:
  using MapNodeOperator_t = std::map<CircleNode *, const circle::Operator *>;

  MapNodeOperator_t _table;
};

/*
 * @brief  CircleNode to circle::Tensor
 *         To find circle::Tensor from CircleNode
 */
class NodeTensorFinder
{
public:
  void enroll(CircleNode *node, const circle::Tensor *tensor);

  const circle::Tensor *tensor(CircleNode *node) const;

private:
  using MapNodeTensor_t = std::map<CircleNode *, const circle::Tensor *>;

  MapNodeTensor_t _table;
};

using TensorIndex = int32_t;

/*
 * @brief  Tensor Index to CircleNode
 *         To find CircleNode from TensorIndex
 */
class IndexNodeFinder
{
public:
  void enroll(TensorIndex idx, CircleNode *node);

  CircleNode *node(TensorIndex idx) const;

private:
  using MapIndexNode_t = std::map<TensorIndex, CircleNode *>;

  MapIndexNode_t _table;
};

class GraphBuilderContext;

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
  virtual void update(GraphBuilderContext *) = 0;
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
  GraphBuilderContext(loco::Graph *g, CircleReader *reader, NodeOpFinder *nofinder,
                      NodeTensorFinder *ntfinder, IndexNodeFinder *infinder, UpdateQueue *updates)
      : _g(g), _reader(reader), _nodeopfinder(nofinder), _nodetensorfinder(ntfinder),
        _indexnodefinder(infinder), _updates(updates)
  {
    // DO NOTHING
  }

  GraphBuilderContext(const GraphBuilderContext &) = delete;
  GraphBuilderContext(GraphBuilderContext &&) = delete;

public:
  loco::Graph *graph() { return _g; }
  CircleReader *reader() { return _reader; }

  NodeOpFinder *opfinder(void) { return _nodeopfinder; }
  NodeTensorFinder *tensorfinder(void) { return _nodetensorfinder; }
  IndexNodeFinder *nodefinder(void) { return _indexnodefinder; }
  UpdateQueue *updates() { return _updates; }

private:
  loco::Graph *_g;
  CircleReader *_reader;
  NodeOpFinder *_nodeopfinder;
  NodeTensorFinder *_nodetensorfinder;
  IndexNodeFinder *_indexnodefinder;
  UpdateQueue *_updates;
};

} // namespace luci

#endif // __LUCI_IMPORT_GRAPH_BUILDER_CONTEXT_H__
