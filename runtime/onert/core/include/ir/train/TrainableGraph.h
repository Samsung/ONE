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
#include "ir/train/ITrainableOperation.h"

namespace onert
{
namespace ir
{
namespace train
{

class TrainableGraph : public IGraph
{
public:
  /**
   * @brief Construct a new Trainable Graph object
   *
   * @param graph
   */
  explicit TrainableGraph();
  explicit TrainableGraph(const TrainableGraph &tgraph);
  explicit TrainableGraph(const Graph &graph);
  ~TrainableGraph() = default;

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
   * @return OperandIndex @c index if successful, UNDEFINED otherwise
   */
  OperandIndex addOperand(OperandIndex index, std::unique_ptr<Operand> &&operand);
  /**
   * @brief Add a new trainable operation to the graph
   *
   * If the given @c operation has at least one invalid operand index, it fails. And @c operation
   * will not be moved so the caller's pointer will be still valid.
   *
   * @param operation Operation to be added
   * @return OperationIndex @c index if successful, UNDEFINED otherwise
   */
  OperationIndex addOperation(std::unique_ptr<ITrainableOperation> &&operation);
  /**
   * @brief Replace a trainable operation which the graph already has
   *
   * If the given @c index is available, it succeeds. And @c operation is moved which invalidates
   * the caller's pointer. If the given @c operation has at least one invalid operand index, it
   * fails. And @c operation will not be moved so the caller's pointer will be still valid.
   *
   * No information in the graph is changed except for replacing an operation.
   *
   * @param operation Operation to be added
   * @return OperationIndex @c index if successful, UNDEFINED otherwise
   */
  OperationIndex replaceOperation(OperationIndex index,
                                  std::unique_ptr<ITrainableOperation> &&operation);

  /**
   * @brief Add a derivative to the graph with the given index and object
   *
   * If the given index is available, it succeeds. And @c derivative is moved which invalidates the
   * caller's pointer. If the given index is already taken, it fails. And @c derivative will not be
   * moved so the caller's pointer will be still valid.
   *
   * @param[in] index      Index to be added
   * @param[in] derivative Derivative operand to be added
   * @return OperandIndex @c index if successful, UNDEFINED otherwise
   */
  OperandIndex addDerivative(OperandIndex index, std::unique_ptr<Operand> &&derivative);

public:
  void changeShape(const OperandIndex &ind, const ir::Shape &new_shape) override;
  void changeDerivativeShape(const OperandIndex &ind, const ir::Shape &new_shape);
  void addInput(const OperandIndex &ind, const std::string &name = "");
  void addOutput(const OperandIndex &ind, const std::string &name = "");
  void verify() const;
  void removeOperand(const OperandIndex &ind);
  void setLayout(Layout layout);
  void setInputs(OperandIndexSequence inputs,
                 std::unordered_map<std::string, IOIndex> name_to_input);
  void setOutputs(OperandIndexSequence outputs,
                  std::unordered_map<std::string, IOIndex> name_to_output);

  // Accessors
public:
  const OperandIndexSequence &getInputs() const override { return _graph.getInputs(); }
  const OperandIndexSequence &getOutputs() const override { return _graph.getOutputs(); }
  IOIndex getInputIndex(const std::string &name) const override;
  IOIndex getOutputIndex(const std::string &name) const override;
  const Operands &operands() const override { return _graph.operands(); }
  Operands &operands() { return _graph.operands(); } // TODO Remove this non-const accessor
  const Operations &operations() const override { return _graph.operations(); }
  const Operands &derivatives() const { return _derivatives; }
  Layout layout() const { return _graph.layout(); }
  const Graph &graph() const { return _graph; }

public:
  const ITrainableOperation &operation(OperationIndex index) const;

public:
  std::vector<ir::OperationIndex> topolSortOperations() const;
  // TODO Support topological sort for backwarding

private:
  Graph _graph;
  Operands _derivatives;
};

} // namespace train
} // namespace ir
} // namespace onert

#endif // __ONERT_IR_TRAIN_TRAINABLE_GRAPH_H__
