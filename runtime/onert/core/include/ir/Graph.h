/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_IR_GRAPH_H__
#define __ONERT_IR_GRAPH_H__

#include <functional>
#include <unordered_map>

#include "ir/IGraph.h"
#include "ir/Model.h"

namespace onert
{
namespace ir
{

struct GraphIO
{
  OperandIndexSequence inputs;
  OperandIndexSequence outputs;
  std::unordered_map<std::string, IOIndex> name_to_input;
  std::unordered_map<std::string, IOIndex> name_to_output;
};

class Graph : public IGraph
{
private:
  enum class Phase
  {
    BUILDING,
    MODEL
  };

public:
  explicit Graph(void);
  explicit Graph(const Graph &);

  ~Graph(void);

  // Graph Building
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
  OperationIndex addOperation(std::unique_ptr<IOperation> &&node);
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
  OperationIndex addOperation(OperationIndex index, std::unique_ptr<IOperation> &&operation);
  /**
   * @brief Replace an operation which the graph already has
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
  OperationIndex replaceOperation(OperationIndex index, std::unique_ptr<IOperation> &&operation);

  void setOperandValue(const OperandIndex &ind, std::shared_ptr<Data> data);
  void changeShape(const OperandIndex &ind, const ir::Shape &new_shape) override;
  void addInput(const OperandIndex &ind, const std::string &name = "");
  void addOutput(const OperandIndex &ind, const std::string &name = "");
  void verify(void) const;
  void removeOperand(const OperandIndex &ind) { _operands.remove(ind); }
  void setLayout(Layout layout) { _layout = layout; }

private:
  bool checkOperandsForOperation(const IOperation &operation);
  void linkOperandToOperation(OperationIndex index, const IOperation &operation);
  void initializeUseDef();
  // TODO Rename to `sweepUnusedOperands`
  // TODO Make this public
  void sweepGarbageOperands();

  // Accessors
public:
  const OperandIndexSequence &getInputs() const override { return _io_info.inputs; }
  OperandIndexSequence &getInputs() { return _io_info.inputs; }
  const OperandIndexSequence &getOutputs() const override { return _io_info.outputs; }
  OperandIndexSequence &getOutputs() { return _io_info.outputs; }
  IOIndex getInputIndex(const std::string &name) const override;
  IOIndex getOutputIndex(const std::string &name) const override;
  const Operands &operands() const override { return _operands; }
  Operands &operands() { return _operands; } // TODO Remove this non-const accessor
  const Operations &operations() const override { return _operations; }
  Operations &operations() { return _operations; }
  Layout layout() const { return _layout; }
  const GraphIO &io_info() const { return _io_info; }

  // Topological sort
public:
  std::vector<ir::OperationIndex> topolSortOperations() const;

private:
  Operations _operations;
  Operands _operands;
  GraphIO _io_info;
  // TFLite and circle's default layout is NHWC;
  Layout _layout{Layout::NHWC};
};

} // namespace ir
} // namespace onert

#endif // __ONERT_IR_GRAPH_H__
