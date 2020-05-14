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

#ifndef __LOCO_IR_NODE_H__
#define __LOCO_IR_NODE_H__

#include "loco/ADT/AnnotatedItem.h"

#include "loco/IR/Use.h"
#include "loco/IR/Dialect.h"
#include "loco/IR/NodePool.forward.h"
#include "loco/IR/Graph.forward.h"
#include "loco/IR/CastHelpers.h"

#include <array>
#include <memory>
#include <set>

namespace loco
{

/**
 * @brief Extensible Node Metadata
 */
struct NodeAnnotation
{
  virtual ~NodeAnnotation() = default;
};

enum class SubstQualifier
{
  Default, // Replace all the occurrences as "Use" (by default)
};

template <SubstQualifier Q> class Subst;

/**
 * @brief Logical unit of computation
 */
class Node : public AnnotatedItem<NodeAnnotation>
{
public:
  friend class Use;
  friend class Subst<SubstQualifier::Default>;
  friend class NodePool;
  friend std::set<Node *> succs(const Node *node);

public:
  Node() = default;

  Node(const Node &) = delete;
  Node(Node &&) = delete;

  virtual ~Node();

public:
  Graph *graph(void) { return _graph; }
  const Graph *graph(void) const { return _graph; }

private:
  /**
   * @brief Set associated "Graph"
   *
   * @note Only "NodePool" class is permitted to invoke this private method.
   */
  void graph(Graph *g) { _graph = g; }

public:
  /**
   * @brief Return "Dialect" identifier that this node belongs to
   *
   * dialect() SHOULD return a valid pointer.
   */
  virtual const Dialect *dialect(void) const = 0;

  virtual uint32_t opnum(void) const = 0;

public:
  /// @brief Return the number of arguments
  virtual uint32_t arity(void) const = 0;

  /// @brief Access N-th argument node
  virtual Node *arg(uint32_t N) const = 0;

  /**
   * @brief Drop all the reference of arguments
   *
   * arg(n) SHOULD return nullptr for every valid n after drop() call.
   */
  virtual void drop(void) = 0;

private:
  /**
   * @brief Associated Graph
   *
   * May be nullptr if no associated Graph exists.
   */
  Graph *_graph = nullptr;

  /**
   * @brief The edges to a node that uses this node as its argument
   *
   * @note "succs" function below accesses this private field.
   */
  std::set<Use *> _uses;
};

/// @brief Enumerate all the predecessors of a given node
std::set<Node *> preds(const Node *node);
/// @brief Enumerate all the successors of a given node
std::set<Node *> succs(const Node *node);

/**
 * @brief A helper for below "replace" helper
 */
template <> class Subst<SubstQualifier::Default>
{
public:
  friend Subst<SubstQualifier::Default> replace(Node *node);

private:
  explicit Subst(Node *from);

public:
  void with(Node *into) const;

private:
  Node *_from;
};

Subst<SubstQualifier::Default> replace(Node *node);

/**
 * @brief A helper dynamic_cast that throws when failed
 */
template <typename T> T must_cast(Node *node) { return _must_cast<T, Node *>(node); }

template <typename T> T must_cast(const Node *node) { return _must_cast<T, const Node *>(node); }

} // namespace loco

#endif // __LOCO_IR_NODE_H__
