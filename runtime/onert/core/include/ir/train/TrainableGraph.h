/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __ONERT_IR_TRAIN_TRAINABLE_GRAPH_H__
#define __ONERT_IR_TRAIN_TRAINABLE_GRAPH_H__

#include <functional>
#include <unordered_map>

#include "ir/Graph.h"
#include "ir/train/TrainableOperations.h"
#include "ir/train/TrainingInfo.h"

namespace onert
{
namespace backend
{
namespace custom
{
class IKernelBuilder;
} // namespace custom
} // namespace backend
} // namespace onert

namespace onert
{
namespace ir
{
namespace train
{

// TODO Find a better relationship with Graph
class TrainableGraph
{
private:
  enum class Phase
  {
    BUILDING,
    MODEL
  };

public:
  /**
   * @brief Construct a new Trainable Graph object
   *
   * @param graph
   */
  explicit TrainableGraph(const Graph &graph);

  ~TrainableGraph(void);

  // TrainableGraph Building
public:
  OperandIndex addOperand(const Shape &shape, const TypeInfo &type);
  /**
   * @brief Add an operand to the graph with the given index and object
   *
   * If the given index is available, it succeeds. And @c operand is moved which invalidates the
   * caller's pointer. If the given index is already taken, it fails. And @c operand will not be
   * moved so the caller's pointer will be still valid.
   *
   * @param[in] index Index to be added
   * @param[in] operand Operand to be added
   * @return OperandIndex @c index if successful, Undefined otherwise
   */
  OperandIndex addOperand(OperandIndex index, std::unique_ptr<Operand> &&operand);
  /**
   * @brief Add an operation to the graph with the given index and object
   *
   * If the given index is available, it succeeds. And @c operation is moved which invalidates the
   * caller's pointer. If the given index is already taken, it fails. And @c operation will not be
   * moved so the caller's pointer will be still valid.
   *
   * @param index Index to be added
   * @param operation Operation to be added
   * @return OperandIndex @c index if successful, Undefined otherwise
   */
  OperationIndex addOperation(OperationIndex index, std::unique_ptr<Operation> &&operation);
  OperationIndex addOperation(std::unique_ptr<Operation> &&node);
  // TODO Find better name
  OperationIndex addTrainableOperation(OperationIndex index,
                                       std::unique_ptr<ITrainableOperation> &&operation);

  void setOperandValue(const OperandIndex &ind, std::shared_ptr<Data> data);
  void verify(void);
  void removeOperand(const OperandIndex &ind) { _graph.removeOperand(ind); }

  // Accessors
public:
  const OperandIndexSequence &getInputs() const { return _graph.getInputs(); }
  const OperandIndexSequence &getOutputs() const { return _graph.getOutputs(); }
  IOIndex getInputIndex(const std::string &name) const;
  IOIndex getOutputIndex(const std::string &name) const;
  const Operands &operands() const { return _graph.operands(); }
  Operands &operands() { return _graph.operands(); } // TODO Remove this non-const accessor
  const Operations &operations() const { return _graph.operations(); }
  Operations &operations() { return _graph.operations(); }
  Layout layout() const { return _graph.layout(); }
  const Graph &graph() const { return _graph; }
  Graph &graph() { return _graph; }

  // For training
  const TrainableOperations &trainable_operations() const { return _operations; }
  TrainableOperations &trainable_operations() { return _operations; }

  // TODO Support topological sort for backwarding

private:
  Graph _graph;
  TrainableOperations _operations;
};

} // namespace train
} // namespace ir
} // namespace onert

#endif // __ONERT_IR_TRAIN_TRAINABLE_GRAPH_H__
