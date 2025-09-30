/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_BACKEND_CPU_VALIDATOR_H__
#define __ONERT_BACKEND_CPU_VALIDATOR_H__

#include "backend/Backend.h"

namespace onert::backend::cpu
{

// TODO Validate operand type
class Validator : public backend::ValidatorBase
{
public:
  virtual ~Validator() = default;
  Validator(const ir::Graph &graph) : backend::ValidatorBase(graph) {}

private:
  void visit(const ir::operation::AddN &) override { _supported = true; }
  void visit(const ir::operation::ArgMinMax &) override { _supported = true; }
  void visit(const ir::operation::BatchMatMul &) override { _supported = true; }
  void visit(const ir::operation::BatchToSpaceND &) override { _supported = true; }
  void visit(const ir::operation::BinaryArithmetic &) override { _supported = true; }
  void visit(const ir::operation::BroadcastTo &) override { _supported = true; }
  void visit(const ir::operation::Comparison &) override { _supported = true; }
  void visit(const ir::operation::Concat &) override { _supported = true; }
  void visit(const ir::operation::Conv2D &) override { _supported = true; }
  void visit(const ir::operation::DepthToSpace &) override { _supported = true; }
  void visit(const ir::operation::DepthwiseConv2D &) override { _supported = true; }
  void visit(const ir::operation::DynamicUpdateSlice &) override { _supported = true; }
  void visit(const ir::operation::ElementwiseActivation &) override { _supported = true; }
  void visit(const ir::operation::ElementwiseBinary &) override { _supported = true; }
  void visit(const ir::operation::ElementwiseUnary &) override { _supported = true; }
  void visit(const ir::operation::ExpandDims &) override { _supported = true; }
  void visit(const ir::operation::Fill &) override { _supported = true; }
  void visit(const ir::operation::FullyConnected &) override { _supported = true; }
  void visit(const ir::operation::FusedBatchNorm &) override { _supported = true; }
  void visit(const ir::operation::Gather &) override { _supported = true; }
  void visit(const ir::operation::L2Normalization &) override { _supported = true; }
  void visit(const ir::operation::LogSoftmax &) override { _supported = true; }
  void visit(const ir::operation::LSTM &) override { _supported = true; }
  void visit(const ir::operation::DetectionPostProcess &) override { _supported = true; }
  void visit(const ir::operation::OneHot &) override { _supported = true; }
  void visit(const ir::operation::Pack &) override { _supported = true; }
  void visit(const ir::operation::Pad &) override { _supported = true; }
  void visit(const ir::operation::Pool2D &) override { _supported = true; }
  void visit(const ir::operation::Pow &) override { _supported = true; }
  void visit(const ir::operation::Range &) override { _supported = true; }
  void visit(const ir::operation::Rank &) override { _supported = true; }
  void visit(const ir::operation::Reduce &) override { _supported = true; }
  void visit(const ir::operation::Reshape &) override { _supported = true; }
  void visit(const ir::operation::ResizeBilinear &) override { _supported = true; }
  void visit(const ir::operation::Reverse &) override { _supported = true; }
  void visit(const ir::operation::RmsNorm &) override { _supported = true; }
  void visit(const ir::operation::RoPE &) override { _supported = true; }
  void visit(const ir::operation::Select &) override { _supported = true; }
  void visit(const ir::operation::Shape &) override { _supported = true; }
  void visit(const ir::operation::Slice &) override { _supported = true; }
  void visit(const ir::operation::Softmax &) override { _supported = true; }
  void visit(const ir::operation::SpaceToBatchND &) override { _supported = true; }
  void visit(const ir::operation::SpaceToDepth &) override { _supported = true; }
  void visit(const ir::operation::Split &) override { _supported = true; }
  void visit(const ir::operation::SplitV &) override { _supported = true; }
  void visit(const ir::operation::SquaredDifference &) override { _supported = true; }
  void visit(const ir::operation::Squeeze &) override { _supported = true; }
  void visit(const ir::operation::StatelessRandomUniform &) override { _supported = true; }
  void visit(const ir::operation::StridedSlice &) override { _supported = true; }
  void visit(const ir::operation::TopKV2 &) override { _supported = true; }
  void visit(const ir::operation::Tile &) override { _supported = true; }
  void visit(const ir::operation::Transpose &) override { _supported = true; }
  void visit(const ir::operation::Unpack &) override { _supported = true; }
};

} // namespace onert::backend::cpu

#endif // __ONERT_BACKEND_CPU_VALIDATOR_H__
