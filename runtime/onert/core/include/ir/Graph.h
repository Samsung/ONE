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

#include "ir/Operands.h"
#include "ir/Operations.h"
#include "ir/Subgraphs.h"

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

class Graph
{
private:
  enum class Phase
  {
    BUILDING,
    MODEL
  };

public:
  Graph(void);
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
  OperationIndex addOperation(std::unique_ptr<Operation> &&node);
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
  void setOperandValue(const OperandIndex &ind, std::shared_ptr<Data> data);
  void addInput(const OperandIndex &ind, const std::string &name = "");
  void addOutput(const OperandIndex &ind, const std::string &name = "");
  void finishBuilding(void);
  void removeOperand(const OperandIndex &ind) { _operands.remove(ind); }
  bool isBuildingPhase(void) const { return _phase == Phase::BUILDING; }
  void setLayout(Layout layout) { _layout = layout; }
  void setSubgraphs(const std::shared_ptr<Subgraphs> &subgs) { _subgraphs = subgs; }

private:
  void initializeUseDef();
  void sweepGarbageOperands();

  // Custom operations support
public:
  void
  bindKernelBuilder(const std::shared_ptr<onert::backend::custom::IKernelBuilder> &kernel_builder)
  {
    _kernel_builder = kernel_builder;
  }

  const std::shared_ptr<backend::custom::IKernelBuilder> &getKernelBuilder() const
  {
    return _kernel_builder;
  }

private:
  std::shared_ptr<backend::custom::IKernelBuilder> _kernel_builder;

  // Accessors
public:
  const OperandIndexSequence &getInputs() const { return _inputs; }
  OperandIndexSequence &getInputs() { return _inputs; }
  const OperandIndexSequence &getOutputs() const { return _outputs; }
  OperandIndexSequence &getOutputs() { return _outputs; }
  IOIndex getInputIndex(const std::string &name) const;
  IOIndex getOutputIndex(const std::string &name) const;
  const Operands &operands() const { return _operands; }
  Operands &operands() { return _operands; } // TODO Remove this non-const accessor
  const Operations &operations() const { return _operations; }
  Operations &operations() { return _operations; }
  const std::shared_ptr<Subgraphs> &subgraphs() const { return _subgraphs; }
  std::shared_ptr<Subgraphs> &subgraphs() { return _subgraphs; }
  Layout layout() const { return _layout; }

  // Topological sort
public:
  std::vector<ir::OperationIndex> topolSortOperations() const;

private:
  Phase _phase{Phase::BUILDING};
  Operations _operations;
  Operands _operands;
  OperandIndexSequence _inputs;
  OperandIndexSequence _outputs;
  std::unordered_map<std::string, IOIndex> _name_to_input;
  std::unordered_map<std::string, IOIndex> _name_to_output;
  // Child subgraphs
  std::shared_ptr<Subgraphs> _subgraphs;
  // TFLite and circle's default layout is NHWC;
  Layout _layout{Layout::NHWC};
};

} // namespace ir
} // namespace onert

#endif // __ONERT_IR_GRAPH_H__
