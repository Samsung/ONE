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

class OperationValidator : public ir::OperationVisitor
{
public:
  OperationValidator(void) = delete;
  OperationValidator(const ir::Graph &graph);

public:
  void operator()();

public:
  void visit(const ir::operation::AddN &node) override;
  void visit(const ir::operation::BatchMatMul &node) override;
  void visit(const ir::operation::BatchToSpaceND &node) override;
  void visit(const ir::operation::BinaryArithmetic &node) override;
  void visit(const ir::operation::Comparison &node) override;
  void visit(const ir::operation::DepthToSpace &node) override;
  void visit(const ir::operation::ElementwiseActivation &node) override;
  void visit(const ir::operation::ElementwiseBinary &node) override;
  void visit(const ir::operation::ElementwiseUnary &node) override;
  void visit(const ir::operation::EmbeddingLookup &node) override;
  void visit(const ir::operation::ExpandDims &node) override;
  void visit(const ir::operation::HashtableLookup &node) override;
  void visit(const ir::operation::Pack &node) override;
  void visit(const ir::operation::Pad &node) override;
  void visit(const ir::operation::ResizeBilinear &node) override;
  void visit(const ir::operation::Reverse &node) override;
  void visit(const ir::operation::Select &node) override;
  void visit(const ir::operation::SpaceToBatchND &node) override;
  void visit(const ir::operation::SpaceToDepth &node) override;
  void visit(const ir::operation::Split &node) override;
  void visit(const ir::operation::SquaredDifference &node) override;
  void visit(const ir::operation::StridedSlice &node) override;
  void visit(const ir::operation::TransposeConv &node) override;
  void visit(const ir::operation::Unpack &node) override;
  void visit(const ir::operation::While &node) override;

private:
  // TODO Remove _ctx field
  const ir::Graph &_graph;
  const ir::Operands &_ctx;
};

} // namespace ir
} // namespace onert

#endif // __ONERT_IR_OPERATION_VALIDATOR_H__
