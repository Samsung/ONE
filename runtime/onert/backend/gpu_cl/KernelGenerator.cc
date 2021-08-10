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
#include <stdexcept>

#include <backend/basic/KernelGeneratorBase.h>

#include "KernelGenerator.h"

#include "ClTensorRegistry.h"
#include "ClFunction.h"
#include "TensorManager.h"

#include "open_cl/selectors/SimpleSelectors.h"

#include "ir/Operations.h"
#include "ir/Operations.Include.h"
#include "ir/Index.h"
#include "ir/DataType.h"
#include "ir/InternalType.h"
#include "exec/NopFunction.h"
#include "exec/FunctionSequence.h"
#include "util/logging.h"
#include "util/Utils.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{

KernelGenerator::KernelGenerator(const ir::Graph &graph,
                                 const std::shared_ptr<TensorBuilder> &tensor_builder,
                                 const std::shared_ptr<ClTensorRegistry<TensorManager>> &tensor_reg,
                                 const std::shared_ptr<CreationContext> &creation_context)
  : basic::KernelGeneratorBase{graph}, _ctx(graph.operands()),
    _operations_ctx(graph.operations()), _current_layout{graph.layout()},
    _tensor_builder(tensor_builder), _tensor_reg(tensor_reg), _creation_context(creation_context)
{
}

std::unique_ptr<exec::FunctionSequence> KernelGenerator::generate(ir::OperationIndex ind)
{
  auto ret = std::make_unique<exec::FunctionSequence>();
  ret->enableDynamicShapeInferer(false);

  const auto &op = _graph.operations().at(ind);
  op.accept(*this);
  ret->append(releaseFunction());
  return ret;
}

void KernelGenerator::visit(const ir::operation::BinaryArithmetic &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto lhs_index{node.getInputs().at(ir::operation::BinaryArithmetic::Input::LHS)};
  const auto rhs_index{node.getInputs().at(ir::operation::BinaryArithmetic::Input::RHS)};

  // const auto activation = node.param().activation;

  OperationDef op_def;
  op_def.precision = CalculationsPrecision::F32;

  op_def.src_tensors.push_back(_tensor_reg->getClTensorReserver(lhs_index)->descriptor);
  auto lhs_shape = _tensor_reg->getClTensorReserver(lhs_index)->shape;

  op_def.src_tensors.push_back(_tensor_reg->getClTensorReserver(rhs_index)->descriptor);
  auto rhs_shape = _tensor_reg->getClTensorReserver(rhs_index)->shape;

  op_def.dst_tensors.push_back(_tensor_reg->getClTensorReserver(ofm_index)->descriptor);
  auto out_shape = _tensor_reg->getClTensorReserver(ofm_index)->shape;

  auto fn = std::make_unique<ClFunction>();

  std::unique_ptr<GPUOperation> gpu_op;
  switch (node.param().arithmetic_type)
  {
    case ir::operation::BinaryArithmetic::ArithmeticType::ADD:
    {
      std::vector<int> channels(2);
      channels[0] = lhs_shape.c;
      channels[1] = rhs_shape.c;
      SelectAdd(op_def, channels, out_shape.c, &gpu_op);

      auto ofm_tensor = _tensor_reg->getClTensor(ofm_index);
      auto lhs_tensor = _tensor_reg->getClTensor(lhs_index);
      auto rhs_tensor = _tensor_reg->getClTensor(rhs_index);
      gpu_op->SetSrc(lhs_tensor->handle(), ir::operation::BinaryArithmetic::Input::LHS);
      gpu_op->SetSrc(rhs_tensor->handle(), ir::operation::BinaryArithmetic::Input::RHS);
      gpu_op->SetDst(ofm_tensor->handle(), 0);

      fn->configure(_creation_context);
      fn->add_operation(std::move(gpu_op));
      break;
    }
    case ir::operation::BinaryArithmetic::ArithmeticType::SUB:
    {
      // NYI
      break;
    }
    case ir::operation::BinaryArithmetic::ArithmeticType::MUL:
    {
      // NYI
      break;
    }
    case ir::operation::BinaryArithmetic::ArithmeticType::DIV:
    {
      // NYI
      break;
    }
    default:
      assert(false && "The BinaryArithmetic operation supports only binary arithmetic operations");
      break;
  }

  _return_fn = std::move(fn);
}
void KernelGenerator::visit(const ir::operation::ResizeBilinear &node)
{
  const auto input_index{node.getInputs().at(ir::operation::ResizeBilinear::Input::INPUT)};
  const auto output_index{node.getOutputs().at(0)};

  Resize2DAttributes attr;
  attr.type = onert::backend::gpu_cl::SamplingType::BILINEAR;
  attr.align_corners = node.param().align_corners;
  attr.half_pixel_centers = node.param().half_pixel_centers;

  if (node.getInputs().size() == 1)
  {
    if (node.param().height_out < 0)
    {
      throw std::runtime_error{
        "ResizeBilinear: size value must be positive value, output_height = " +
        std::to_string(node.param().height_out)};
    }
    if (node.param().width_out < 0)
    {
      throw std::runtime_error{
        "ResizeBilinear: size value must be positive value, output_width = " +
        std::to_string(node.param().width_out)};
    }
    attr.new_shape = HW(node.param().height_out, node.param().width_out);
  }
  else
  {
    assert(node.getInputs().size() == 2);
    const auto size_index{node.getInputs().at(ir::operation::ResizeBilinear::Input::SIZE)};
    auto size_vec = _ctx.at(size_index).asVector<int32_t>();
    if (size_vec[0] < 0)
    {
      throw std::runtime_error{
        "ResizeBilinear: size value must be positive value, output_height = " +
        std::to_string(size_vec[0])};
    }
    if (size_vec[1] < 0)
    {
      throw std::runtime_error{
        "ResizeBilinear: size value must be positive value, output_width = " +
        std::to_string(size_vec[1])};
    }
    attr.new_shape = HW(size_vec[0], size_vec[1]);
  }

  OperationDef op_def;
  op_def.precision = CalculationsPrecision::F32;

  op_def.src_tensors.push_back(_tensor_reg->getClTensorReserver(input_index)->descriptor);
  op_def.dst_tensors.push_back(_tensor_reg->getClTensorReserver(output_index)->descriptor);

  auto fn = std::make_unique<ClFunction>();

  std::unique_ptr<GPUOperation> gpu_op;
  SelectResize(op_def, attr, &gpu_op);

  auto input_tensor = _tensor_reg->getClTensor(input_index);
  auto output_tensor = _tensor_reg->getClTensor(output_index);

  gpu_op->SetSrc(input_tensor->handle(), ir::operation::ResizeBilinear::Input::INPUT);
  gpu_op->SetDst(output_tensor->handle(), 0);

  fn->configure(_creation_context);
  fn->add_operation(std::move(gpu_op));

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::ResizeNearestNeighbor &node)
{
  const auto input_index{node.getInputs().at(ir::operation::ResizeNearestNeighbor::Input::INPUT)};
  const auto output_index{node.getOutputs().at(0)};

  Resize2DAttributes attr;
  attr.type = onert::backend::gpu_cl::SamplingType::NEAREST;
  attr.align_corners = node.param().align_corners;

  if (node.getInputs().size() == 1)
  {
    if (node.param().height_out < 0)
    {
      throw std::runtime_error{
        "ResizeNearestNeighbor: size value must be positive value, output_height = " +
        std::to_string(node.param().height_out)};
    }
    if (node.param().width_out < 0)
    {
      throw std::runtime_error{
        "ResizeNearestNeighbor: size value must be positive value, output_width = " +
        std::to_string(node.param().width_out)};
    }
    attr.new_shape = HW(node.param().height_out, node.param().width_out);
  }
  else
  {
    assert(node.getInputs().size() == 2);
    const auto size_index{node.getInputs().at(ir::operation::ResizeNearestNeighbor::Input::SIZE)};
    auto size_vec = _ctx.at(size_index).asVector<int32_t>();
    if (size_vec[0] < 0)
    {
      throw std::runtime_error{
        "ResizeNearestNeighbor: size value must be positive value, output_height = " +
        std::to_string(size_vec[0])};
    }
    if (size_vec[1] < 0)
    {
      throw std::runtime_error{
        "ResizeNearestNeighbor: size value must be positive value, output_width = " +
        std::to_string(size_vec[1])};
    }
    attr.new_shape = HW(size_vec[0], size_vec[1]);
  }

  OperationDef op_def;
  op_def.precision = CalculationsPrecision::F32;

  op_def.src_tensors.push_back(_tensor_reg->getClTensorReserver(input_index)->descriptor);
  op_def.dst_tensors.push_back(_tensor_reg->getClTensorReserver(output_index)->descriptor);

  auto fn = std::make_unique<ClFunction>();

  std::unique_ptr<GPUOperation> gpu_op;
  SelectResize(op_def, attr, &gpu_op);

  auto input_tensor = _tensor_reg->getClTensor(input_index);
  auto output_tensor = _tensor_reg->getClTensor(output_index);

  gpu_op->SetSrc(input_tensor->handle(), ir::operation::ResizeNearestNeighbor::Input::INPUT);
  gpu_op->SetDst(output_tensor->handle(), 0);

  fn->configure(_creation_context);
  fn->add_operation(std::move(gpu_op));

  _return_fn = std::move(fn);
}

} // namespace gpu_cl
} // namespace backend
} // namespace onert
