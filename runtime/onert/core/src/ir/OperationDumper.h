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
  void visit(const operation::Abs &) override;
  void visit(const operation::Add &node) override;
  void visit(const operation::ArgMax &) override;
  void visit(const operation::AvgPool2D &node) override;
  void visit(const operation::BatchToSpaceND &node) override;
  void visit(const operation::BroadcastTo &) override;
  void visit(const operation::Cast &) override;
  void visit(const operation::Comparison &) override;
  void visit(const operation::Concat &node) override;
  void visit(const operation::Conv2D &node) override;
  void visit(const operation::ConvertFp16ToFp32 &node) override;
  void visit(const operation::ConvertFp32ToFp16 &node) override;
  void visit(const operation::Cos &node) override;
  void visit(const operation::DepthToSpace &) override;
  void visit(const operation::DepthwiseConv2D &node) override;
  void visit(const operation::Dequantize &) override;
  void visit(const operation::Div &) override;
  void visit(const operation::EmbeddingLookup &) override;
  void visit(const operation::Exp &) override;
  void visit(const operation::ExpandDims &) override;
  void visit(const operation::Floor &) override;
  void visit(const operation::FullyConnected &node) override;
  void visit(const operation::Gather &) override;
  void visit(const operation::HashtableLookup &) override;
  void visit(const operation::InstanceNorm &) override;
  void visit(const operation::L2Normalization &) override;
  void visit(const operation::L2Pool2D &) override;
  void visit(const operation::LocalResponseNormalization &) override;
  void visit(const operation::Log &) override;
  void visit(const operation::LogicalAnd &) override;
  void visit(const operation::LogicalNot &) override;
  void visit(const operation::LogicalOr &) override;
  void visit(const operation::Logistic &) override;
  void visit(const operation::LSTM &) override;
  void visit(const operation::MaxPool2D &node) override;
  void visit(const operation::Mean &) override;
  void visit(const operation::Mul &) override;
  void visit(const operation::Neg &) override;
  void visit(const operation::Pack &) override;
  void visit(const operation::Pad &) override;
  void visit(const operation::Permute &node) override;
  void visit(const operation::Pow &node) override;
  void visit(const operation::PReLU &) override;
  void visit(const operation::Range &) override;
  void visit(const operation::ReduceAll &) override;
  void visit(const operation::ReduceAny &) override;
  void visit(const operation::ReduceMax &) override;
  void visit(const operation::ReduceMin &) override;
  void visit(const operation::ReduceSum &) override;
  void visit(const operation::ReduceProd &) override;
  void visit(const operation::ReLU &) override;
  void visit(const operation::ReLU1 &) override;
  void visit(const operation::ReLU6 &) override;
  void visit(const operation::Reshape &node) override;
  void visit(const operation::ResizeBilinear &) override;
  void visit(const operation::Reverse &) override;
  void visit(const operation::RNN &) override;
  void visit(const operation::Round &) override;
  void visit(const operation::RSQRT &) override;
  void visit(const operation::Select &node) override;
  void visit(const operation::Shape &node) override;
  void visit(const operation::Sin &node) override;
  void visit(const operation::Softmax &node) override;
  void visit(const operation::SpaceToBatchND &) override;
  void visit(const operation::SpaceToDepth &) override;
  void visit(const operation::Split &) override;
  void visit(const operation::SQRT &) override;
  void visit(const operation::SquaredDifference &) override;
  void visit(const operation::Squeeze &) override;
  void visit(const operation::Slice &) override;
  void visit(const operation::StridedSlice &) override;
  void visit(const operation::Sub &) override;
  void visit(const operation::Tanh &) override;
  void visit(const operation::Tile &) override;
  void visit(const operation::TopKV2 &) override;
  void visit(const operation::TransposeConv &) override;
  void visit(const operation::Transpose &) override;
  void visit(const operation::Unpack &) override;
  void visit(const operation::Min &) override;
  void visit(const operation::Max &) override;
  void visit(const operation::OneHot &) override;
  void visit(const operation::If &) override;
  void visit(const operation::While &) override;
  void visit(const operation::ZerosLike &) override;
};

} // namespace ir
} // namespace onert

#endif // __ONERT_OPERATION_DUMPER_H__
