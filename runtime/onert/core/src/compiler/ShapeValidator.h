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

#ifndef __ONERT_COMPILER_SHAPE_VALIDATOR_H__
#define __ONERT_COMPILER_SHAPE_VALIDATOR_H__

#include "ir/Layout.h"
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
namespace compiler
{

class ShapeValidator : public ir::OperationVisitor
{
public:
  ShapeValidator(void) = delete;
  ShapeValidator(const ir::Graph &graph);
  ShapeValidator(const ShapeValidator &) = delete;
  ShapeValidator(ShapeValidator &&) = delete;
  ~ShapeValidator() = default;

public:
  ShapeValidator &operator=(const ShapeValidator &) = delete;
  ShapeValidator &operator=(ShapeValidator &&) = delete;
  void operator()();

public:
  void visit(const ir::operation::BatchMatMul &node) override;
  void visit(const ir::operation::BatchToSpaceND &node) override;
  void visit(const ir::operation::BCQFullyConnected &node) override;
  void visit(const ir::operation::BCQGather &node) override;
  void visit(const ir::operation::Conv2D &node) override;
  void visit(const ir::operation::Comparison &node) override;
  void visit(const ir::operation::DepthwiseConv2D &node) override;
  void visit(const ir::operation::FullyConnected &node) override;
  void visit(const ir::operation::Softmax &node) override;
  void visit(const ir::operation::InstanceNorm &node) override;
  void visit(const ir::operation::Permute &node) override;
  void visit(const ir::operation::Pool2D &node) override;
  void visit(const ir::operation::Reduce &node) override;
  void visit(const ir::operation::Transpose &node) override;
  void visit(const ir::operation::RNN &node) override;
  void visit(const ir::operation::SpaceToBatchND &node) override;
  void visit(const ir::operation::SpaceToDepth &node) override;
  void visit(const ir::operation::ElementwiseActivation &node) override;
  void visit(const ir::operation::ElementwiseBinary &node) override;
  void visit(const ir::operation::ElementwiseUnary &node) override;
  void visit(const ir::operation::EmbeddingLookup &node) override;
  void visit(const ir::operation::ExpandDims &node) override;
  void visit(const ir::operation::HashtableLookup &node) override;
  void visit(const ir::operation::TransposeConv &node) override;
  void visit(const ir::operation::Gather &node) override;
  void visit(const ir::operation::DepthToSpace &node) override;
  void visit(const ir::operation::Pack &node) override;
  void visit(const ir::operation::LSTM &node) override;
  void visit(const ir::operation::L2Normalization &node) override;
  void visit(const ir::operation::Unpack &node) override;
  void visit(const ir::operation::Pad &node) override;
  void visit(const ir::operation::Select &node) override;
  void visit(const ir::operation::StridedSlice &node) override;
  void visit(const ir::operation::Split &node) override;
  void visit(const ir::operation::Shape &node) override;
  void visit(const ir::operation::ResizeBilinear &node) override;
  void visit(const ir::operation::Reverse &node) override;
  void visit(const ir::operation::If &node) override;
  void visit(const ir::operation::While &node) override;
  void visit(const ir::operation::SquaredDifference &node) override;
  void visit(const ir::operation::Tile &node) override;
  void visit(const ir::operation::Range &node) override;
  void visit(const ir::operation::MatrixBandPart &node) override;
  void visit(const ir::operation::LogSoftmax &node) override;

private:
  void checkUnaryOp(const ir::Operation &node);

private:
  const ir::Graph &_graph;
};

} // namespace compiler
} // namespace onert

#endif // __ONERT_COMPILER_SHAPE_VALIDATOR_H__
