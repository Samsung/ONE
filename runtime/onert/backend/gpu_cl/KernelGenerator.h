/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_BACKEND_GPU_CL_KERNEL_GENERATOR_H__
#define __ONERT_BACKEND_GPU_CL_KERNEL_GENERATOR_H__

#include "TensorRegistry.h"
#include "backend/basic/TensorRegistry.h"
#include "TensorBuilder.h"
#include "TensorManager.h"

#include "tensorflow/lite/delegates/gpu/api.h"

#include <backend/CustomKernelBuilder.h>
#include <backend/basic/KernelGeneratorBase.h>
#include <backend/BackendContext.h>
#include <ir/Operands.h>
#include <ir/Operations.h>
#include <ir/Operations.Include.h>

namespace onert
{
namespace backend
{
namespace gpu_cl
{

class KernelGenerator : public basic::KernelGeneratorBase
{
public:
  KernelGenerator(const ir::Graph &graph, const std::shared_ptr<TensorBuilder> &tensor_builder,
                  const std::shared_ptr<TensorRegistry> &tensor_reg,
                  const std::shared_ptr<tflite::gpu::cl::CreationContext> &creation_context);

  std::unique_ptr<exec::FunctionSequence> generate(ir::OperationIndex ind) override;

  void get_operation(FunctionMap &Functions);

private:
  void visit(const ir::operation::BinaryArithmetic &) override;
  void visit(const ir::operation::Conv2D &) override;
  void visit(const ir::operation::DepthwiseConv2D &) override;
  void visit(const ir::operation::ElementwiseActivation &) override;
  void visit(const ir::operation::Pool2D &) override;
  void visit(const ir::operation::Reshape &) override;
  void visit(const ir::operation::Softmax &) override;
  absl::Status readConstTensor(const ir::OperandIndex &index, tflite::gpu::TensorOrScalar *param);
  absl::Status readConstTensor(
    const ir::OperandIndex &index,
    absl::variant<tflite::gpu::Tensor<tflite::gpu::Linear, tflite::gpu::DataType::FLOAT32>,
                  tflite::gpu::Tensor<tflite::gpu::HWC, tflite::gpu::DataType::FLOAT32>> *alpha);
  void addClNode(const std::vector<ir::OperandIndex> &inputs,
                 const std::vector<ir::OperandIndex> &outputs,
                 std::unique_ptr<tflite::gpu::GPUOperation> gpu_op);

private:
  const ir::Operands &_ctx;
  const ir::Operations &_operations_ctx;
  ir::Layout _current_layout;
  std::shared_ptr<TensorBuilder> _tensor_builder;
  std::shared_ptr<TensorRegistry> _tensor_reg;
  std::shared_ptr<tflite::gpu::cl::CreationContext> _creation_context;
  std::vector<tflite::gpu::cl::CLNode> _nodes;
  ir::OperationIndex _operation_index;
  std::vector<ir::OperationIndex> _operation_indexes;
};

} // namespace gpu_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_GPU_CL_KERNEL_GENERATOR_H__
