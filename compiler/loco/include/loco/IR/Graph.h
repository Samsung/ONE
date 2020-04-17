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

#ifndef __LOCO_IR_GRAPH_H__
#define __LOCO_IR_GRAPH_H__

#include "loco/IR/DataType.h"
// TODO Include "Node.h" instead
#include "loco/IR/Nodes.h"
#include "loco/IR/NodePool.h"
#include "loco/IR/GraphInputIndex.h"
#include "loco/IR/GraphOutputIndex.h"

#include "loco/ADT/ObjectPool.h"

#include <initializer_list>
#include <set>
#include <string>
#include <memory>
#include <vector>

namespace loco
{

// TODO Introduce Named trait
enum class Trait
{
  // Any "DataTyped" class has the following methods
  // - DataType dtype(void) const;
  // - void dtype(const DataType &value);
  DataTyped,
  // Any "TensorShaped" class has the following methods
  // - const TensorShape *shape(void) const;
  // - void shape(std::unique_ptr<TensorShape> &&);
  // - void shape(std::initializer_list<Dimension> &&);
  //
  // TODO Rename NodeMixin::TensorShape as NodeMixin::NDShape
  TensorShaped,
};

template <Trait T> class Mixin;

// TODO Re-implement NodeMixin<NodeTrait::DataType> using this mixin
template <> class Mixin<Trait::DataTyped>
{
public:
  Mixin() = default;

public:
  const DataType &dtype(void) const { return _dtype; }
  void dtype(const DataType &value) { _dtype = value; }

private:
  DataType _dtype = DataType::Unknown;
};

template <> class Mixin<Trait::TensorShaped>
{
public:
  Mixin() = default;

public:
  const TensorShape *shape(void) const { return _shape.get(); }
  void shape(std::unique_ptr<TensorShape> &&shape) { _shape = std::move(shape); }
  void shape(std::initializer_list<Dimension> dims);

private:
  std::unique_ptr<TensorShape> _shape = nullptr;
};

/**
 * @brief Trait for elements with name
 */
class NamedEntity
{
public:
  const std::string &name(void) const { return _name; }
  void name(const std::string &name) { _name = name; }

/// If new interface methods are added to this class they also will need to
/// be added in `using` of this macro to get them visible from inherited classes
#define LOCO_NAMED_ENTITY_EXPOSE using NamedEntity::name

private:
  std::string _name;
};

/**
 * @brief Graph-level Input Metadata
 */
class GraphInput final : private NamedEntity,
                         public Mixin<Trait::DataTyped>,
                         public Mixin<Trait::TensorShaped>
{
public:
  LOCO_NAMED_ENTITY_EXPOSE;

  // TODO Use GraphInputIndex (instead of uint32_t)
  GraphInput(uint32_t index) : _index{index}
  {
    // DO NOTHING
  }

  GraphInput(const GraphInput &) = delete;
  GraphInput(GraphInput &&) = delete;

  ~GraphInput() = default;

public:
  GraphInputIndex index(void) const { return _index; }

private:
  uint32_t _index;
};

/**
 * @brief Graph-level Output Metadata
 */
class GraphOutput final : private NamedEntity,
                          public Mixin<Trait::DataTyped>,
                          public Mixin<Trait::TensorShaped>
{
public:
  LOCO_NAMED_ENTITY_EXPOSE;

  // TODO Use GraphOutputIndex (instead of uint32_t)
  GraphOutput(uint32_t index) : _index{index}
  {
    // DO NOTHING
  }

  GraphOutput(const GraphOutput &) = delete;
  GraphOutput(GraphOutput &&) = delete;

  ~GraphOutput() = default;

public:
  GraphOutputIndex index(void) const { return _index; }

private:
  uint32_t _index;
};

/**
 * @brief A neural network graph
 */
class Graph final : public NamedEntity
{
public:
  /**
   * @brief Node Pool
   *
   * This alias confines the impact of changes to loco internals.
   *
   * TODO Remove this alias
   */
  using NodeContext = NodePool;

  /**
   * @brief Object Pool with Simple Factory Method
   *
   * TODO Remove this unused class
   */
  template <typename T> struct SimpleFactoryObjectPool : public ObjectPool<T>
  {
    virtual ~SimpleFactoryObjectPool() = default;

    T *create(void)
    {
      std::unique_ptr<T> ptr{new T};
      return ObjectPool<T>::take(std::move(ptr));
    }
  };

  /**
   * @brief GraphInput Pool
   */
  struct InputContext final : public ObjectPool<GraphInput>
  {
    GraphInput *create(void);
  };

  /**
   * @brief GraphOutput Pool
   */
  struct OutputContext final : public ObjectPool<GraphOutput>
  {
    GraphOutput *create(void);
  };

public:
  Graph()
  {
    // Associate "NodeContext" and the current "Graph"
    _node_ctx.graph(this);
  }

  // Copy/Move is not allowed for Graph
  Graph(const Graph &) = delete;
  Graph(Graph &&) = delete;

  ~Graph() = default;

public:
  NodeContext *nodes(void) { return &_node_ctx; }
  const NodeContext *nodes(void) const { return &_node_ctx; }
  InputContext *inputs(void) { return &_input_ctx; }
  const InputContext *inputs(void) const { return &_input_ctx; }
  OutputContext *outputs(void) { return &_output_ctx; }
  const OutputContext *outputs(void) const { return &_output_ctx; }

private:
  NodeContext _node_ctx;
  InputContext _input_ctx;
  OutputContext _output_ctx;
};

struct GraphInputIndexQueryService : public DialectService
{
  virtual ~GraphInputIndexQueryService() = default;

  /**
   * @brief Check whether a given node is associated with any Graph-level input
   */
  virtual bool associated(const Node *node) const = 0;

  /**
   * Exceptions
   * - index SHOULD throw std::invalid_argument exception if a given node is not associated with
   *   any input (i.e. assocaited above returns false).
   */
  virtual GraphInputIndex index(const Node *node) const = 0;
};

std::vector<Node *> input_nodes(const Graph *);

struct GraphOutputIndexQueryService : public DialectService
{
  virtual ~GraphOutputIndexQueryService() = default;

  /**
   * @brief Check whether a given node is associated with any Graph-level output
   */
  virtual bool associated(const Node *node) const = 0;

  /**
   * Exceptions
   * - index SHOULD throw std::invalid_argument exception if a given node is not associated with
   *   any output (i.e. assocaited above returns false).
   */
  virtual GraphOutputIndex index(const Node *node) const = 0;
};

// TODO Use "const Graph *"
std::vector<Node *> output_nodes(Graph *);

/**
 * @brief Enumerate all the nodes in a given graph
 *
 * NOTE This method returns std::set<Node *> unlike input_nodes and output_nodes.
 *
 * Please use traverse algorithms that "Algorithm.h" provides (such as postorder_traversal)
 * if order is relevant for implementation.
 */
std::set<Node *> all_nodes(Graph *);

std::unique_ptr<Graph> make_graph(void);

} // namespace loco

#endif // __LOCO_IR_GRAPH_H__
