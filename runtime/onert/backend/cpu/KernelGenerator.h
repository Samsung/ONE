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

#ifndef __ONERT_BACKEND_CPU_KERNEL_GENERATOR_H__
#define __ONERT_BACKEND_CPU_KERNEL_GENERATOR_H__

#include "TensorBuilder.h"
#include "Tensor.h"

#include <backend/CustomKernelBuilder.h>
#include <backend/IKernelGenerator.h>
#include <ir/Operands.h>
#include <ir/Operations.h>

namespace onert
{
namespace backend
{
namespace cpu
{

class KernelGenerator : public IKernelGenerator
{
public:
  KernelGenerator(const ir::Operands &operands_ctx, const ir::Operations &operations_ctx,
                  const std::shared_ptr<TensorBuilder> &tensor_builder,
                  const std::shared_ptr<custom::IKernelBuilder> &kernel_builder);

  using IKernelGenerator::visit;

  void visit(const ir::OpSequence &) override;
  void visit(const ir::operation::Conv2D &) override;
  void visit(const ir::operation::DepthwiseConv2D &) override;
  void visit(const ir::operation::MaxPool2D &) override;
  void visit(const ir::operation::AvgPool2D &) override;
  void visit(const ir::operation::Concat &) override;
  void visit(const ir::operation::Fill &) override;
  void visit(const ir::operation::FullyConnected &) override;
  void visit(const ir::operation::Reshape &) override;
  void visit(const ir::operation::Squeeze &) override;
  void visit(const ir::operation::Softmax &) override;
  void visit(const ir::operation::Comparison &) override;
  void visit(const ir::operation::Add &) override;
  void visit(const ir::operation::Sub &) override;
  void visit(const ir::operation::Mul &) override;
  void visit(const ir::operation::Div &) override;
  void visit(const ir::operation::Einsum &) override;
  void visit(const ir::operation::Gather &) override;
  void visit(const ir::operation::Custom &node) override;
  void visit(const ir::operation::Exp &) override;
  void visit(const ir::operation::ExpandDims &) override;
  void visit(const ir::operation::Logistic &) override;
  void visit(const ir::operation::Pad &) override;
  void visit(const ir::operation::Max &) override;
  void visit(const ir::operation::Min &) override;
  void visit(const ir::operation::Tanh &) override;
  void visit(const ir::operation::Pack &) override;
  void visit(const ir::operation::Unpack &) override;
  void visit(const ir::operation::OneHot &) override;
  void visit(const ir::operation::Cast &) override;
  void visit(const ir::operation::Transpose &) override;
  void visit(const ir::operation::ReduceSum &) override;
  void visit(const ir::operation::ReduceAll &) override;
  void visit(const ir::operation::ReduceAny &) override;
  void visit(const ir::operation::ReduceMax &) override;
  void visit(const ir::operation::ReduceMin &) override;
  void visit(const ir::operation::ReLU &) override;
  void visit(const ir::operation::Select &) override;
  void visit(const ir::operation::Slice &) override;
  void visit(const ir::operation::StridedSlice &) override;
  void visit(const ir::operation::Split &) override;
  void visit(const ir::operation::Abs &) override;
  void visit(const ir::operation::Cos &) override;
  void visit(const ir::operation::Sin &) override;
  void visit(const ir::operation::RSQRT &) override;
  void visit(const ir::operation::Shape &) override;
  void visit(const ir::operation::ReduceProd &) override;
  void visit(const ir::operation::Reverse &) override;
  void visit(const ir::operation::Neg &) override;
  void visit(const ir::operation::ArgMax &) override;
  void visit(const ir::operation::Mean &) override;
  void visit(const ir::operation::Log &) override;
  void visit(const ir::operation::Round &) override;
  void visit(const ir::operation::Pow &) override;
  void visit(const ir::operation::LogicalNot &) override;
  void visit(const ir::operation::ZerosLike &) override;
  void visit(const ir::operation::SquaredDifference &) override;
  void visit(const ir::operation::Tile &) override;
  void visit(const ir::operation::LogicalOr &) override;
  void visit(const ir::operation::Range &) override;
  void visit(const ir::operation::MatrixBandPart &) override;
  void visit(const ir::operation::BatchMatMul &) override;
  void visit(const ir::operation::BroadcastTo &) override;
  void visit(const ir::operation::FusedBatchNorm &) override;

private:
  const ir::Operands &_ctx;
  const ir::Operations &_operations_ctx;
  std::shared_ptr<TensorBuilder> _tensor_builder;
  std::shared_ptr<backend::custom::IKernelBuilder> _kernel_builder;
  ir::Layout _current_op_seq_layout;
};

} // namespace cpu
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_CPU_KERNEL_GENERATOR_H__
