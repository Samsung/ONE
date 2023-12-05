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

#include "ExternalContext.h"
#include "TensorBuilder.h"
#include "backend/basic/TensorRegistry.h"
#include "Tensor.h"

#include <backend/CustomKernelBuilder.h>
#include <backend/basic/KernelGeneratorBase.h>
#include <ir/Operands.h>
#include <ir/Operations.h>

namespace onert
{
namespace backend
{
namespace cpu
{

class KernelGenerator : public basic::KernelGeneratorBase
{
public:
  KernelGenerator(const ir::Graph &graph, const std::shared_ptr<TensorBuilder> &tensor_builder,
                  const std::shared_ptr<basic::TensorRegistry> &tensor_reg,
                  const std::shared_ptr<custom::IKernelBuilder> &kernel_builder,
                  const std::shared_ptr<ExternalContext> &external_context);

  std::unique_ptr<exec::FunctionSequence> generate(ir::OperationIndex op_ind) override;

  void visit(const ir::operation::AddN &) override;
  void visit(const ir::operation::ArgMinMax &) override;
  void visit(const ir::operation::BatchMatMul &) override;
  void visit(const ir::operation::BatchToSpaceND &) override;
  void visit(const ir::operation::BinaryArithmetic &) override;
  void visit(const ir::operation::BroadcastTo &) override;
  void visit(const ir::operation::Comparison &) override;
  void visit(const ir::operation::Concat &) override;
  void visit(const ir::operation::Conv2D &) override;
  void visit(const ir::operation::Custom &node) override;
  void visit(const ir::operation::DepthToSpace &) override;
  void visit(const ir::operation::DepthwiseConv2D &) override;
  void visit(const ir::operation::Einsum &) override;
  void visit(const ir::operation::ElementwiseActivation &) override;
  void visit(const ir::operation::ElementwiseBinary &) override;
  void visit(const ir::operation::ElementwiseUnary &) override;
  void visit(const ir::operation::ExpandDims &) override;
  void visit(const ir::operation::Fill &) override;
  void visit(const ir::operation::FullyConnected &) override;
  void visit(const ir::operation::FusedBatchNorm &) override;
  void visit(const ir::operation::Gather &) override;
  void visit(const ir::operation::L2Normalization &) override;
  void visit(const ir::operation::LogSoftmax &) override;
  void visit(const ir::operation::LSTM &) override;
  void visit(const ir::operation::MatrixBandPart &) override;
  void visit(const ir::operation::DetectionPostProcess &) override;
  void visit(const ir::operation::OneHot &) override;
  void visit(const ir::operation::Pack &) override;
  void visit(const ir::operation::Pad &) override;
  void visit(const ir::operation::Pool2D &) override;
  void visit(const ir::operation::Pow &) override;
  void visit(const ir::operation::Range &) override;
  void visit(const ir::operation::Rank &) override;
  void visit(const ir::operation::Reduce &) override;
  void visit(const ir::operation::Reshape &) override;
  void visit(const ir::operation::ResizeBilinear &node) override;
  void visit(const ir::operation::Reverse &) override;
  void visit(const ir::operation::Select &) override;
  void visit(const ir::operation::Shape &) override;
  void visit(const ir::operation::Slice &) override;
  void visit(const ir::operation::Softmax &) override;
  void visit(const ir::operation::SpaceToBatchND &) override;
  void visit(const ir::operation::SpaceToDepth &) override;
  void visit(const ir::operation::Split &) override;
  void visit(const ir::operation::SplitV &) override;
  void visit(const ir::operation::SquaredDifference &) override;
  void visit(const ir::operation::Squeeze &) override;
  void visit(const ir::operation::StatelessRandomUniform &) override;
  void visit(const ir::operation::StridedSlice &) override;
  void visit(const ir::operation::Tile &) override;
  void visit(const ir::operation::Transpose &) override;
  void visit(const ir::operation::Unpack &) override;

private:
  const ir::Operands &_ctx;
  const ir::Operations &_operations_ctx;
  std::shared_ptr<TensorBuilder> _tensor_builder;
  std::shared_ptr<basic::TensorRegistry> _tensor_reg;
  std::shared_ptr<backend::custom::IKernelBuilder> _kernel_builder;
  const std::shared_ptr<ExternalContext> _external_context;
};

} // namespace cpu
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_CPU_KERNEL_GENERATOR_H__
