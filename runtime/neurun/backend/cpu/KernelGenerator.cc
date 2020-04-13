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

#include "KernelGenerator.h"

#include "kernel/AddLayer.h"
#include "kernel/AvgPoolLayer.h"
#include "kernel/CastLayer.h"
#include "kernel/CompareLayer.h"
#include "kernel/ConcatLayer.h"
#include "kernel/ConvolutionLayer.h"
#include "kernel/DepthwiseConvolutionLayer.h"
#include "kernel/DivLayer.h"
#include "kernel/ExpLayer.h"
#include "kernel/FullyConnectedLayer.h"
#include "kernel/GatherLayer.h"
#include "kernel/LogisticLayer.h"
#include "kernel/MaxLayer.h"
#include "kernel/MaxPoolLayer.h"
#include "kernel/MinLayer.h"
#include "kernel/MulLayer.h"
#include "kernel/OneHotLayer.h"
#include "kernel/OperationUtils.h"
#include "kernel/PackLayer.h"
#include "kernel/PadLayer.h"
#include "kernel/PermuteLayer.h"
#include "kernel/ReduceLayer.h"
#include "kernel/ReshapeLayer.h"
#include "kernel/SliceLayer.h"
#include "kernel/SoftMaxLayer.h"
#include "kernel/SubLayer.h"
#include "kernel/TanhLayer.h"
#include "kernel/TransposeLayer.h"
#include "kernel/UnpackLayer.h"

#include <backend/Backend.h>
#include <backend/IConfig.h>
#include <memory>
#include <util/Utils.h>
#include <util/logging.h>

#include <stdexcept>

namespace neurun
{
namespace backend
{
namespace cpu
{

KernelGenerator::KernelGenerator(
    const ir::Operands &operand_ctx, const std::shared_ptr<TensorBuilder> &tensor_builder,
    const std::shared_ptr<backend::custom::IKernelBuilder> &kernel_builer)
    : _ctx(operand_ctx), _tensor_builder(tensor_builder), _kernel_builder(kernel_builer),
      _current_op_seq_layout(ir::Layout::UNKNOWN)
{
  // DO NOTHING
}

void KernelGenerator::visit(const ir::OpSequence &op_seq)
{
  // TODO Move this to IKernelGenerator
  //      (all derivatives have the same implementation for this)
  assert(!_return_fn_seq);
  _return_fn_seq = std::make_unique<exec::FunctionSequence>();
  _current_op_seq_layout = op_seq.getLayout();
  for (const auto &e : op_seq.operations())
  {
    const auto &node = *(e.node);
    node.accept(*this);
    _return_fn_seq->append(releaseFunction());

    // NOTE Permute node has tensors of the other backends
    if (node.opcode() != ir::OpCode::Permute)
    {
      for (const auto &ind : node.getInputs() + node.getOutputs())
      {
        auto tensor = _tensor_builder->at(ind);
        if (tensor)
        {
          tensor->increase_ref();
        }
      }
    }
  }
}

void KernelGenerator::visit(const ir::operation::Conv2D &node)
{
  using ir::operation::Conv2D;

  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(Conv2D::Input::INPUT)};
  const auto ker_index{node.getInputs().at(Conv2D::Input::KERNEL)};
  const auto bias_index{node.getInputs().at(Conv2D::Input::BIAS)};

  const auto stride = node.param().stride;
  const auto ifm_shape = _ctx.at(ifm_index).shape().asFeature(_current_op_seq_layout);
  const auto ofm_shape = _ctx.at(ofm_index).shape().asFeature(_current_op_seq_layout);
  // Kernel format is [depth_out, kernel_height, kernel_width, depth_in].
  const auto &ker_shape = _ctx.at(ker_index).shape();
  const auto ker_height = ker_shape.dim(1);
  const auto ker_width = ker_shape.dim(2);
  const auto padding_type = node.param().padding.type;
  const auto padding = ir::calculatePadding(node.param().padding, ifm_shape, ofm_shape, stride,
                                            ker_width, ker_height);
  const auto activation = node.param().activation;

  auto ofm_alloc = _tensor_builder->at(ofm_index).get();
  auto ifm_alloc = _tensor_builder->at(ifm_index).get();
  auto ker_alloc = _tensor_builder->at(ker_index).get();
  auto bias_alloc = _tensor_builder->at(bias_index).get();

  auto fn = std::make_unique<::neurun::backend::cpu::kernel::ConvolutionLayer>();

  fn->configure(ifm_alloc, ker_alloc, bias_alloc, padding_type, padding.left, padding.right,
                padding.top, padding.bottom, stride.horizontal, stride.vertical, activation,
                ofm_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::DepthwiseConv2D &node)
{
  using ir::operation::DepthwiseConv2D;

  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(DepthwiseConv2D::Input::INPUT)};
  const auto ker_index{node.getInputs().at(DepthwiseConv2D::Input::KERNEL)};
  const auto bias_index{node.getInputs().at(DepthwiseConv2D::Input::BIAS)};

  const auto stride = node.param().stride;
  const auto ifm_shape = _ctx.at(ifm_index).shape().asFeature(_current_op_seq_layout);
  const auto ofm_shape = _ctx.at(ofm_index).shape().asFeature(_current_op_seq_layout);
  // Kernel format is [1, kernel_height, kernel_width, depth_out].
  const auto &ker_shape = _ctx.at(ker_index).shape();
  const auto ker_height = ker_shape.dim(1);
  const auto ker_width = ker_shape.dim(2);
  const auto padding = ir::calculatePadding(node.param().padding, ifm_shape, ofm_shape, stride,
                                            ker_width, ker_height);
  const auto multiplier = node.param().multiplier;
  const auto activation = node.param().activation;

  auto ofm_alloc = _tensor_builder->at(ofm_index).get();
  auto ifm_alloc = _tensor_builder->at(ifm_index).get();
  auto ker_alloc = _tensor_builder->at(ker_index).get();
  auto bias_alloc = _tensor_builder->at(bias_index).get();

  auto fn = std::make_unique<::neurun::backend::cpu::kernel::DepthwiseConvolutionLayer>();

  fn->configure(ifm_alloc, ker_alloc, bias_alloc, padding.left, padding.right, padding.top,
                padding.bottom, stride.horizontal, stride.vertical, multiplier, activation,
                ofm_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::MaxPool2D &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::MaxPool2D::Input::INPUT)};

  const auto kh = node.param().kh;
  const auto kw = node.param().kw;

  const auto stride = node.param().stride;
  const auto ifm_shape = _ctx.at(ifm_index).shape().asFeature(_current_op_seq_layout);
  const auto ofm_shape = _ctx.at(ofm_index).shape().asFeature(_current_op_seq_layout);
  const auto padding =
      ir::calculatePadding(node.param().padding, ifm_shape, ofm_shape, stride, kw, kh);
  const auto activation = node.param().activation;

  auto ofm_alloc = _tensor_builder->at(ofm_index).get();
  auto ifm_alloc = _tensor_builder->at(ifm_index).get();

  auto fn = std::make_unique<::neurun::backend::cpu::kernel::MaxPoolLayer>();

  fn->configure(ifm_alloc, padding.left, padding.right, padding.top, padding.bottom,
                stride.horizontal, stride.vertical, kw, kh, activation, ofm_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::AvgPool2D &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::AvgPool2D::Input::INPUT)};

  const auto kh = node.param().kh;
  const auto kw = node.param().kw;
  const auto stride = node.param().stride;
  const auto ifm_shape = _ctx.at(ifm_index).shape().asFeature(_current_op_seq_layout);
  const auto ofm_shape = _ctx.at(ofm_index).shape().asFeature(_current_op_seq_layout);
  const auto padding =
      ir::calculatePadding(node.param().padding, ifm_shape, ofm_shape, stride, kw, kh);
  const auto activation = node.param().activation;

  auto ofm_alloc = _tensor_builder->at(ofm_index).get();
  auto ifm_alloc = _tensor_builder->at(ifm_index).get();

  auto fn = std::make_unique<::neurun::backend::cpu::kernel::AvgPoolLayer>();

  fn->configure(ifm_alloc, padding.left, padding.right, padding.top, padding.bottom,
                stride.horizontal, stride.vertical, kw, kh, activation, ofm_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Concat &node)
{
  const auto ofm_index{node.getOutputs().at(0)};

  const auto rank = _ctx.at(ofm_index).shape().rank();
  const auto axis =
      ::neurun::backend::cpu::kernel::getAxis(rank, node.param().axis, _current_op_seq_layout);

  auto output_alloc = _tensor_builder->at(ofm_index).get();

  std::vector<const operand::Tensor *> input_tensors;
  for (auto &ifm_idx : node.getInputs())
    input_tensors.emplace_back(_tensor_builder->at(ifm_idx).get());

  auto fn = std::make_unique<::neurun::backend::cpu::kernel::ConcatLayer>();

  fn->configure(input_tensors, axis, output_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::FullyConnected &node)
{
  using ir::operation::FullyConnected;

  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(FullyConnected::Input::INPUT)};
  const auto weight_index{node.getInputs().at(FullyConnected::Input::WEIGHT)};
  const auto bias_index{node.getInputs().at(FullyConnected::Input::BIAS)};
  const auto activation = node.param().activation;

  auto output_alloc = _tensor_builder->at(output_index).get();
  auto input_alloc = _tensor_builder->at(input_index).get();
  auto weight_alloc = _tensor_builder->at(weight_index).get();
  auto bias_alloc = _tensor_builder->at(bias_index).get();

  auto fn = std::make_unique<::neurun::backend::cpu::kernel::FullyConnectedLayer>();

  fn->configure(input_alloc, weight_alloc, bias_alloc, activation, output_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Reshape &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Reshape::Input::INPUT)};

  auto output_alloc = _tensor_builder->at(output_index).get();
  auto input_alloc = _tensor_builder->at(input_index).get();

  auto fn = std::make_unique<::neurun::backend::cpu::kernel::ReshapeLayer>();

  fn->configure(input_alloc, output_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Squeeze &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Squeeze::Input::INPUT)};

  auto output_alloc = _tensor_builder->at(output_index).get();
  auto input_alloc = _tensor_builder->at(input_index).get();

  // Squeeze can share same kernel with reshape
  auto fn = std::make_unique<::neurun::backend::cpu::kernel::ReshapeLayer>();

  fn->configure(input_alloc, output_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Softmax &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Softmax::Input::INPUT)};

  const auto beta = node.param().beta;

  auto output_alloc = _tensor_builder->at(output_index).get();
  auto input_alloc = _tensor_builder->at(input_index).get();

  auto fn = std::make_unique<::neurun::backend::cpu::kernel::SoftMaxLayer>();

  fn->configure(input_alloc, beta, output_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Add &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto lhs_index{node.getInputs().at(ir::operation::Add::Input::LHS)};
  const auto rhs_index{node.getInputs().at(ir::operation::Add::Input::RHS)};

  const auto activation = node.param().activation;

  auto ofm_alloc = _tensor_builder->at(ofm_index).get();
  auto lhs_alloc = _tensor_builder->at(lhs_index).get();
  auto rhs_alloc = _tensor_builder->at(rhs_index).get();

  auto fn = std::make_unique<::neurun::backend::cpu::kernel::AddLayer>();

  fn->configure(lhs_alloc, rhs_alloc, activation, ofm_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Comparison &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto lhs_index{node.getInputs().at(ir::operation::Comparison::Input::INPUT0)};
  const auto rhs_index{node.getInputs().at(ir::operation::Comparison::Input::INPUT1)};

  auto ofm_alloc = _tensor_builder->at(ofm_index).get();
  auto lhs_alloc = _tensor_builder->at(lhs_index).get();
  auto rhs_alloc = _tensor_builder->at(rhs_index).get();

  auto comparison_type = node.param().comparison_type;

  auto fn = std::make_unique<::neurun::backend::cpu::kernel::CompareLayer>();

  fn->configure(lhs_alloc, rhs_alloc, comparison_type, ofm_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Gather &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Gather::Input::INPUT)};
  const auto indices_index{node.getInputs().at(ir::operation::Gather::Input::INDICES)};

  auto output_alloc = _tensor_builder->at(output_index).get();
  auto input_alloc = _tensor_builder->at(input_index).get();
  auto indices_alloc = _tensor_builder->at(indices_index).get();

  const auto backend_layout = output_alloc->layout();
  UNUSED_RELEASE(backend_layout);

  // NOTE The frontend layout and backend layout must be the same for this operation.
  //      If not the same, we have to add a stage(?) to perform permutation of output tensor. It
  //      is not not efficient even if it works well. If so, it would be better to set the
  //      layout of these backend tensors to the same layout.
  //      There is also one thing we have to think about. This operation depends on the layout of
  //      a model. For example, if a model in NHWC has this operation as output rank == 4, indices
  //      rank == 2 and axis == 2, this operation should work as the axis W and C, but the axis W
  //      and C are not sequential in NCHW. So the backend in NCHW cannot handle this case.
  assert(backend_layout == input_alloc->layout());
  assert(backend_layout == indices_alloc->layout());
  const auto &input_shape = _ctx.at(input_index).shape();
  UNUSED_RELEASE(input_shape);
  assert(input_shape.rank() < 4 || _current_op_seq_layout == backend_layout);

  const auto axis_raw = node.param().axis;
  const auto axis_value = (axis_raw < 0 ? (input_shape.rank() + axis_raw) : axis_raw);

  auto fn = std::make_unique<::neurun::backend::cpu::kernel::GatherLayer>();

  fn->configure(input_alloc, indices_alloc, output_alloc, axis_value);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Sub &node)
{
  // The same as Add
  const auto ofm_index{node.getOutputs().at(0)};
  const auto lhs_index{node.getInputs().at(ir::operation::Sub::Input::LHS)};
  const auto rhs_index{node.getInputs().at(ir::operation::Sub::Input::RHS)};

  const auto activation = node.param().activation;

  auto ofm_alloc = _tensor_builder->at(ofm_index).get();
  auto lhs_alloc = _tensor_builder->at(lhs_index).get();
  auto rhs_alloc = _tensor_builder->at(rhs_index).get();

  auto fn = std::make_unique<::neurun::backend::cpu::kernel::SubLayer>();

  fn->configure(lhs_alloc, rhs_alloc, activation, ofm_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Mul &node)
{
  // The same as Add
  const auto ofm_index{node.getOutputs().at(0)};
  const auto lhs_index{node.getInputs().at(ir::operation::Sub::Input::LHS)};
  const auto rhs_index{node.getInputs().at(ir::operation::Sub::Input::RHS)};

  const auto activation = node.param().activation;

  auto ofm_alloc = _tensor_builder->at(ofm_index).get();
  auto lhs_alloc = _tensor_builder->at(lhs_index).get();
  auto rhs_alloc = _tensor_builder->at(rhs_index).get();

  auto fn = std::make_unique<::neurun::backend::cpu::kernel::MulLayer>();

  fn->configure(lhs_alloc, rhs_alloc, activation, ofm_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::OneHot &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto indices_index{node.getInputs().at(ir::operation::OneHot::INDICES)};

  const auto depth = node.param().depth;
  const auto on_value = node.param().on_value;
  const auto off_value = node.param().off_value;
  const auto axis = node.param().axis;

  auto output_alloc = _tensor_builder->at(output_index).get();
  auto indices_alloc = _tensor_builder->at(indices_index).get();

  assert(indices_alloc->data_type() == OperandType::INT32);
  assert(axis <= static_cast<int>(indices_alloc->num_dimensions()));

  auto fn = std::make_unique<::neurun::backend::cpu::kernel::OneHotLayer>();

  fn->configure(indices_alloc, output_alloc, depth, on_value, off_value, axis);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Div &node)
{
  // The same as Add
  const auto ofm_index{node.getOutputs().at(0)};
  const auto lhs_index{node.getInputs().at(ir::operation::Sub::Input::LHS)};
  const auto rhs_index{node.getInputs().at(ir::operation::Sub::Input::RHS)};

  const auto activation = node.param().activation;

  auto ofm_alloc = _tensor_builder->at(ofm_index).get();
  auto lhs_alloc = _tensor_builder->at(lhs_index).get();
  auto rhs_alloc = _tensor_builder->at(rhs_index).get();

  auto fn = std::make_unique<::neurun::backend::cpu::kernel::DivLayer>();

  fn->configure(lhs_alloc, rhs_alloc, activation, ofm_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Permute &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(0)};

  const auto &shape = _ctx.at(output_index).shape();
  const auto input_backend_ctx = node.param().input_backend_ctx;
  const auto output_backend_ctx = node.param().output_backend_ctx;
  const auto data_type = node.getDataType();

  auto output_tensor = output_backend_ctx->tensor_builder->tensorAt(output_index);
  auto input_tensor = input_backend_ctx->tensor_builder->tensorAt(input_index);
  assert(output_tensor != nullptr);
  assert(input_tensor != nullptr);

  auto fn = std::make_unique<::neurun::backend::cpu::kernel::PermuteLayer>();

  // TODO Support NCHW frontend
  auto out_shape = shape;
  if (shape.rank() == 4 && output_tensor->layout() == ir::Layout::NCHW)
  {
    out_shape.dim(1) = shape.dim(3);
    out_shape.dim(2) = shape.dim(1);
    out_shape.dim(3) = shape.dim(2);
  }

  const auto permute_type = node.getPermuteType();
  // Check Permutation Type
  const auto inferPermuteType = [&]() {
    if (input_tensor->layout() == ir::Layout::NHWC && output_tensor->layout() == ir::Layout::NCHW)
    {
      return ir::operation::Permute::Type::NHWC_TO_NCHW;
    }
    else if (input_tensor->layout() == ir::Layout::NCHW &&
             output_tensor->layout() == ir::Layout::NHWC)
    {
      return ir::operation::Permute::Type::NCHW_TO_NHWC;
    }
    else
    {
      return ir::operation::Permute::Type::COPY;
    }
  }();
  UNUSED_RELEASE(inferPermuteType);
  assert(permute_type == inferPermuteType);

  fn->configure(input_tensor, output_tensor, out_shape, permute_type, data_type);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Custom &node)
{
  auto get_type_info = [this](const ir::Operand &operand) -> custom::TypeInfo {
    const auto &frontend_shape = operand.shape();
    custom::Shape shape(frontend_shape.rank());
    for (auto d = 0; d < frontend_shape.rank(); ++d)
    {
      shape.dim(d) = frontend_shape.dim(d);
    }

    return {shape, operand.typeInfo().type()};
  };

  auto fill_op_info = [&](const ir::OperandIndexSequence &opSeq,
                          std::vector<custom::TypeInfo> &types, std::vector<void *> &allocs) {
    for (auto &idx : opSeq)
    {
      const auto &operand = _ctx.at(idx);
      // TODO make sure using `_current_op_seq_layout` is correct for custom operations
      types.emplace_back(get_type_info(operand));
      auto in_alloc = _tensor_builder->at(idx)->buffer();
      allocs.emplace_back(in_alloc);
    }
  };

  backend::custom::CustomKernelConfigParams params{};

  fill_op_info(node.getInputs(), params.input_types, params.input_allocations);
  fill_op_info(node.getOutputs(), params.output_types, params.output_allocations);

  params.userdata = node.userdata().data;
  params.userdata_size = node.userdata().size;

  auto fn = _kernel_builder->buildKernel(node.id(), std::move(params));

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Exp &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Exp::Input::INPUT)};

  auto output_alloc = _tensor_builder->at(output_index).get();
  auto input_alloc = _tensor_builder->at(input_index).get();

  auto fn = std::make_unique<::neurun::backend::cpu::kernel::ExpLayer>();

  fn->configure(input_alloc, output_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Logistic &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Logistic::Input::INPUT)};

  auto output_alloc = _tensor_builder->at(output_index).get();
  auto input_alloc = _tensor_builder->at(input_index).get();

  auto fn = std::make_unique<::neurun::backend::cpu::kernel::LogisticLayer>();

  fn->configure(input_alloc, output_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Tanh &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Tanh::Input::INPUT)};

  auto output_alloc = _tensor_builder->at(output_index).get();
  auto input_alloc = _tensor_builder->at(input_index).get();

  auto fn = std::make_unique<::neurun::backend::cpu::kernel::TanhLayer>();

  fn->configure(input_alloc, output_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Pack &node)
{
  const auto ofm_index{node.getOutputs().at(0)};

  const auto rank = node.param().rank;
  const auto axis =
      ::neurun::backend::cpu::kernel::getAxis(rank, node.param().axis, _current_op_seq_layout);

  assert(-rank <= axis && axis < rank);

  auto output_alloc = _tensor_builder->at(ofm_index).get();

  std::vector<const operand::Tensor *> input_tensors;
  for (auto &ifm_idx : node.getInputs())
    input_tensors.emplace_back(_tensor_builder->at(ifm_idx).get());

  auto fn = std::make_unique<::neurun::backend::cpu::kernel::PackLayer>();

  fn->configure(input_tensors, axis, output_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Unpack &node)
{
  const auto input_index{node.getInputs().at(0)};

  const auto rank = node.param().rank;
  const auto axis =
      ::neurun::backend::cpu::kernel::getAxis(rank, node.param().axis, _current_op_seq_layout);

  assert(-rank <= axis && axis < rank);

  auto input_alloc = _tensor_builder->at(input_index).get();

  std::vector<operand::Tensor *> output_tensors;
  for (auto &output_idx : node.getOutputs())
    output_tensors.emplace_back(_tensor_builder->at(output_idx).get());

  auto fn = std::make_unique<::neurun::backend::cpu::kernel::UnpackLayer>();

  uint32_t axis_resolved = (axis < 0 ? axis + rank : axis);

  fn->configure(input_alloc, axis_resolved, node.param().num, output_tensors);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Pad &node)
{
  const auto input_index{node.getInputs().at(ir::operation::Pad::Input::INPUT)};
  const auto pad_index{node.getInputs().at(ir::operation::Pad::Input::PAD)};
  const auto output_index{node.getOutputs().at(0)};
  assert(_ctx.at(pad_index).data());

  auto input = _tensor_builder->at(input_index).get();
  auto output = _tensor_builder->at(output_index).get();
  auto pad_rank = _ctx.at(pad_index).shape().dim(0);
  auto pad_base = reinterpret_cast<const int32_t *>(_ctx.at(pad_index).data()->base());

  auto fn = std::make_unique<::neurun::backend::cpu::kernel::PadLayer>();

  fn->configure(input, output, pad_base, pad_rank);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Max &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto lhs_index{node.getInputs().at(ir::operation::Max::Input::LHS)};
  const auto rhs_index{node.getInputs().at(ir::operation::Max::Input::RHS)};

  auto ofm_alloc = _tensor_builder->at(ofm_index).get();
  auto lhs_alloc = _tensor_builder->at(lhs_index).get();
  auto rhs_alloc = _tensor_builder->at(rhs_index).get();

  auto fn = std::make_unique<::neurun::backend::cpu::kernel::MaxLayer>();

  fn->configure(lhs_alloc, rhs_alloc, ofm_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Min &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto lhs_index{node.getInputs().at(ir::operation::Min::Input::LHS)};
  const auto rhs_index{node.getInputs().at(ir::operation::Min::Input::RHS)};

  auto ofm_alloc = _tensor_builder->at(ofm_index).get();
  auto lhs_alloc = _tensor_builder->at(lhs_index).get();
  auto rhs_alloc = _tensor_builder->at(rhs_index).get();

  auto fn = std::make_unique<::neurun::backend::cpu::kernel::MinLayer>();

  fn->configure(lhs_alloc, rhs_alloc, ofm_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Cast &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::Cast::Input::INPUT)};

  auto ofm_alloc = _tensor_builder->at(ofm_index).get();
  auto ifm_alloc = _tensor_builder->at(ifm_index).get();

  auto fn = std::make_unique<::neurun::backend::cpu::kernel::CastLayer>();

  fn->configure(ifm_alloc, ofm_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Transpose &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(0)};

  auto output_alloc = _tensor_builder->at(output_index).get();
  auto input_alloc = _tensor_builder->at(input_index).get();
  auto rank = node.param().rank;

  auto fn = std::make_unique<::neurun::backend::cpu::kernel::TransposeLayer>();

  fn->configure(input_alloc, output_alloc, node.param().perm, rank);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::ReduceSum &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(0)};

  auto output_alloc = _tensor_builder->at(output_index).get();
  auto input_alloc = _tensor_builder->at(input_index).get();

  auto fn = std::make_unique<kernel::ReduceLayer>();

  fn->configure(input_alloc, output_alloc, kernel::ReduceType::kSum, node.param().axes,
                node.param().keep_dims);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::ReduceMax &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(0)};

  auto output_alloc = _tensor_builder->at(output_index).get();
  auto input_alloc = _tensor_builder->at(input_index).get();

  auto fn = std::make_unique<kernel::ReduceLayer>();

  fn->configure(input_alloc, output_alloc, kernel::ReduceType::kMax, node.param().axes,
                node.param().keep_dims);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::ReduceMin &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(0)};

  auto output_alloc = _tensor_builder->at(output_index).get();
  auto input_alloc = _tensor_builder->at(input_index).get();

  auto fn = std::make_unique<kernel::ReduceLayer>();

  fn->configure(input_alloc, output_alloc, kernel::ReduceType::kMin, node.param().axes,
                node.param().keep_dims);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Slice &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Slice::Input::INPUT)};
  const auto begins_index{node.getInputs().at(ir::operation::Slice::Input::BEGINS)};
  const auto sizes_index{node.getInputs().at(ir::operation::Slice::Input::SIZES)};

  auto output_alloc = _tensor_builder->at(output_index).get();
  auto input_alloc = _tensor_builder->at(input_index).get();
  auto begins_alloc = _tensor_builder->at(begins_index).get();
  auto sizes_alloc = _tensor_builder->at(sizes_index).get();

  auto fn = std::make_unique<::neurun::backend::cpu::kernel::SliceLayer>();

  fn->configure(input_alloc, begins_alloc, sizes_alloc, output_alloc);

  _return_fn = std::move(fn);
}

} // namespace cpu
} // namespace backend
} // namespace neurun
