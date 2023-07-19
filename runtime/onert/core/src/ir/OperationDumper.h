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

#ifndef __ONERT_OPERATION_DUMPER_H__
#define __ONERT_OPERATION_DUMPER_H__

#include "ir/OperationVisitor.h"
#include <string>

namespace onert
{
namespace ir
{

class OperationDumper : public OperationVisitor
{
public:
  OperationDumper(const std::string &start_msg);

public:
  void visit(const operation::ArgMinMax &) override;
  void visit(const operation::BatchToSpaceND &node) override;
  void visit(const operation::BCQFullyConnected &node) override;
  void visit(const operation::BinaryArithmetic &node) override;
  void visit(const operation::BroadcastTo &) override;
  void visit(const operation::Comparison &) override;
  void visit(const operation::Concat &node) override;
  void visit(const operation::Conv2D &node) override;
  void visit(const operation::ConvertFp16ToFp32 &node) override;
  void visit(const operation::ConvertFp32ToFp16 &node) override;
  void visit(const operation::DepthToSpace &) override;
  void visit(const operation::DepthwiseConv2D &node) override;
  void visit(const operation::ElementwiseActivation &) override;
  void visit(const operation::ElementwiseBinary &) override;
  void visit(const operation::ElementwiseUnary &) override;
  void visit(const operation::EmbeddingLookup &) override;
  void visit(const operation::ExpandDims &) override;
  void visit(const operation::Fill &) override;
  void visit(const operation::FullyConnected &node) override;
  void visit(const operation::Gather &) override;
  void visit(const operation::HashtableLookup &) override;
  void visit(const operation::InstanceNorm &) override;
  void visit(const operation::L2Normalization &) override;
  void visit(const operation::LocalResponseNormalization &) override;
  void visit(const operation::Loss &node) override;
  void visit(const operation::LSTM &) override;
  void visit(const operation::Pack &) override;
  void visit(const operation::Pad &) override;
  void visit(const operation::Permute &node) override;
  void visit(const operation::Pool2D &node) override;
  void visit(const operation::Pow &node) override;
  void visit(const operation::PReLU &) override;
  void visit(const operation::Range &) override;
  void visit(const operation::Rank &) override;
  void visit(const operation::Reduce &) override;
  void visit(const operation::Reshape &node) override;
  void visit(const operation::ResizeBilinear &) override;
  void visit(const operation::ResizeNearestNeighbor &) override;
  void visit(const operation::Reverse &) override;
  void visit(const operation::RNN &) override;
  void visit(const operation::Select &node) override;
  void visit(const operation::Shape &node) override;
  void visit(const operation::Softmax &node) override;
  void visit(const operation::SpaceToBatchND &) override;
  void visit(const operation::SpaceToDepth &) override;
  void visit(const operation::Split &) override;
  void visit(const operation::SquaredDifference &) override;
  void visit(const operation::Squeeze &) override;
  void visit(const operation::Slice &) override;
  void visit(const operation::StridedSlice &) override;
  void visit(const operation::StatelessRandomUniform &) override;
  void visit(const operation::Tile &) override;
  void visit(const operation::TopKV2 &) override;
  void visit(const operation::TransposeConv &) override;
  void visit(const operation::Transpose &) override;
  void visit(const operation::Unpack &) override;
  void visit(const operation::OneHot &) override;
  void visit(const operation::If &) override;
  void visit(const operation::While &) override;
};

} // namespace ir
} // namespace onert

#endif // __ONERT_OPERATION_DUMPER_H__
