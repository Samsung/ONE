/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_IR_OPERATION_VALIDATOR_H__
#define __ONERT_IR_OPERATION_VALIDATOR_H__

#include "ir/OperationVisitor.h"
#include "ir/Operations.h"
#include "ir/Operands.h"

namespace onert
{
namespace ir
{
class Graph;
class Operands;
} // namespace ir
} // namespace onert

namespace onert
{
namespace ir
{

class OperationValidator : public OperationVisitor
{
public:
  OperationValidator(void) = delete;
  OperationValidator(const Graph &graph);

public:
  void operator()();

public:
  void visit(const operation::AddN &node) override;
  void visit(const operation::ArgMinMax &node) override;
  void visit(const operation::BatchMatMul &node) override;
  void visit(const operation::BatchToSpaceND &node) override;
  void visit(const operation::BinaryArithmetic &node) override;
  void visit(const operation::Comparison &node) override;
  void visit(const operation::Concat &node) override;
  void visit(const operation::Conv2D &node) override;
  void visit(const operation::CumSum &node) override;
  void visit(const operation::DepthToSpace &node) override;
  void visit(const operation::DepthwiseConv2D &node) override;
  void visit(const operation::DetectionPostProcess &node) override;
  void visit(const operation::ElementwiseActivation &node) override;
  void visit(const operation::ElementwiseBinary &node) override;
  void visit(const operation::ElementwiseUnary &node) override;
  void visit(const operation::EmbeddingLookup &node) override;
  void visit(const operation::ExpandDims &node) override;
  void visit(const operation::Fill &node) override;
  void visit(const operation::HashtableLookup &node) override;
  void visit(const operation::Pack &node) override;
  void visit(const operation::Pad &node) override;
  void visit(const operation::Rank &node) override;
  void visit(const operation::ResizeBilinear &node) override;
  void visit(const operation::Reverse &node) override;
  void visit(const operation::Select &node) override;
  void visit(const operation::Shape &node) override;
  void visit(const operation::Slice &node) override;
  void visit(const operation::Softmax &node) override;
  void visit(const operation::SpaceToBatchND &node) override;
  void visit(const operation::SpaceToDepth &node) override;
  void visit(const operation::Split &node) override;
  void visit(const operation::SquaredDifference &node) override;
  void visit(const operation::StatelessRandomUniform &node) override;
  void visit(const operation::StridedSlice &node) override;
  void visit(const operation::TransposeConv &node) override;
  void visit(const operation::Unpack &node) override;
  void visit(const operation::While &node) override;

private:
  DataType operandType(const OperandIndex &idx);
  bool isConstant(const OperandIndex &idx);
  bool isSameType(const OperandIndex &idx1, const OperandIndex &idx2);
  bool isSameQuantParam(const OperandIndex &idx1, const OperandIndex &idx2);
  bool isValidType(const OperandIndex &idx, const DataType &type);
  bool isValidType(const OperandIndex &idx, std::initializer_list<DataType> valid_types);

private:
  const Operations &_operations;
  const Operands &_operands;
};

} // namespace ir
} // namespace onert

#endif // __ONERT_IR_OPERATION_VALIDATOR_H__
