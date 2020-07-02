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

#include "ops/AbsLayer.h"
#include "ops/AddLayer.h"
#include "ops/ArgMinMaxLayer.h"
#include "ops/AvgPoolLayer.h"
#include "ops/CastLayer.h"
#include "ops/CompareLayer.h"
#include "ops/ConcatLayer.h"
#include "ops/ConvolutionLayer.h"
#include "ops/CosLayer.h"
#include "ops/DepthwiseConvolutionLayer.h"
#include "ops/DivLayer.h"
#include "ops/EinsumLayer.h"
#include "ops/ExpLayer.h"
#include "ops/ExpandDimsLayer.h"
#include "ops/FillLayer.h"
#include "ops/FullyConnectedLayer.h"
#include "ops/GatherLayer.h"
#include "ops/LogLayer.h"
#include "ops/LogisticLayer.h"
#include "ops/MaxLayer.h"
#include "ops/MaxPoolLayer.h"
#include "ops/MeanLayer.h"
#include "ops/MinLayer.h"
#include "ops/MulLayer.h"
#include "ops/NegLayer.h"
#include "ops/OneHotLayer.h"
#include "ops/OperationUtils.h"
#include "ops/PackLayer.h"
#include "ops/PadLayer.h"
#include "ops/PowLayer.h"
#include "ops/RangeLayer.h"
#include "ops/ReduceLayer.h"
#include "ops/ReLULayer.h"
#include "ops/ReshapeLayer.h"
#include "ops/ReverseLayer.h"
#include "ops/RoundLayer.h"
#include "ops/RsqrtLayer.h"
#include "ops/SelectLayer.h"
#include "ops/ShapeLayer.h"
#include "ops/SinLayer.h"
#include "ops/SliceLayer.h"
#include "ops/SoftMaxLayer.h"
#include "ops/StridedSliceLayer.h"
#include "ops/SplitLayer.h"
#include "ops/SubLayer.h"
#include "ops/TanhLayer.h"
#include "ops/TileLayer.h"
#include "ops/TransposeLayer.h"
#include "ops/UnpackLayer.h"
#include "ops/LogicalNotLayer.h"
#include "ops/ZerosLikeLayer.h"
#include "ops/SquaredDiffLayer.h"
#include "ops/LogicalOrLayer.h"
#include "ops/MatrixBandPartLayer.h"
#include "ops/BatchMatMulLayer.h"
#include "ops/BroadcastToLayer.h"
#include "ops/FusedBatchNormLayer.h"

#include <backend/Backend.h>
#include <backend/IConfig.h>
#include <memory>
#include <util/Utils.h>
#include <util/logging.h>

#include <stdexcept>

namespace onert
{
namespace backend
{
namespace cpu
{

KernelGenerator::KernelGenerator(
    const ir::Operands &operands_ctx, const ir::Operations &operations_ctx,
    const std::shared_ptr<TensorBuilder> &tensor_builder,
    const std::shared_ptr<backend::custom::IKernelBuilder> &kernel_builder)
    : _ctx(operands_ctx), _operations_ctx{operations_ctx}, _tensor_builder(tensor_builder),
      _kernel_builder(kernel_builder), _current_op_seq_layout(ir::Layout::UNKNOWN)
{
  // DO NOTHING
}

void KernelGenerator::visit(const ir::OpSequence &op_seq)
{
  assert(!_return_fn_seq);
  assert(_tensor_builder->dynamicTensorManager());
  assert(_tensor_builder->tensorRegistry());

  auto dyn_tensor_manager = _tensor_builder->dynamicTensorManager();
  auto dyn_shape_inferer = std::make_unique<shape_inference::DynamicInferer>(
      _ctx, dyn_tensor_manager, _tensor_builder->tensorRegistry());

  // TODO Always returning FunctionSequenceForDynamicBackend may cause performance issue
  _return_fn_seq = std::make_unique<exec::FunctionSequenceForDynamicBackend>(
      op_seq, _operations_ctx, std::move(dyn_shape_inferer), dyn_tensor_manager);

  _current_op_seq_layout = op_seq.getLayout();
  for (const auto &operation_idx : op_seq.operations())
  {
    const auto &node = _operations_ctx.at(operation_idx);
    node.accept(*this);
    _return_fn_seq->append(releaseFunction());

    for (const auto &ind : (node.getInputs() | ir::Remove::UNDEFINED) + node.getOutputs())
    {
      auto portable_tensor = _tensor_builder->portableAt(ind);
      if (portable_tensor)
      {
        assert(portable_tensor->layout() == ir::Layout::NHWC);
      }

      auto tensor = _tensor_builder->at(ind);
      if (tensor)
      {
        tensor->increase_ref();
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

  auto ofm_alloc = _tensor_builder->portableAt(ofm_index).get();
  auto ifm_alloc = _tensor_builder->portableAt(ifm_index).get();
  auto ker_alloc = _tensor_builder->portableAt(ker_index).get();
  auto bias_alloc = _tensor_builder->portableAt(bias_index).get();

  const auto stride = node.param().stride;
  const auto activation = node.param().activation;
  const auto param_padding = node.param().padding;
  auto fn = std::make_unique<ops::ConvolutionLayer>();

  if (_ctx.at(ifm_index).info().isDynamic() || _ctx.at(ker_index).info().isDynamic())
  {
    fn->configure(ifm_alloc, ker_alloc, bias_alloc, param_padding.type, param_padding.param.left,
                  param_padding.param.right, param_padding.param.top, param_padding.param.bottom,
                  stride.horizontal, stride.vertical, activation, ofm_alloc);

    _return_fn = std::move(fn);
    return;
  }
  const auto ifm_shape = _ctx.at(ifm_index).shape().asFeature(_current_op_seq_layout);
  const auto ofm_shape = _ctx.at(ofm_index).shape().asFeature(_current_op_seq_layout);
  // Kernel format is [depth_out, kernel_height, kernel_width, depth_in].
  const auto &ker_shape = _ctx.at(ker_index).shape();
  const auto ker_height = ker_shape.dim(1);
  const auto ker_width = ker_shape.dim(2);

  const auto padding =
      ir::calculatePadding(param_padding, ifm_shape, ofm_shape, stride, ker_width, ker_height);

  fn->configure(ifm_alloc, ker_alloc, bias_alloc, param_padding.type, padding.left, padding.right,
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

  auto ofm_alloc = _tensor_builder->portableAt(ofm_index).get();
  auto ifm_alloc = _tensor_builder->portableAt(ifm_index).get();
  auto ker_alloc = _tensor_builder->portableAt(ker_index).get();
  auto bias_alloc = _tensor_builder->portableAt(bias_index).get();

  auto fn = std::make_unique<ops::DepthwiseConvolutionLayer>();

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

  auto ofm_alloc = _tensor_builder->portableAt(ofm_index).get();
  auto ifm_alloc = _tensor_builder->portableAt(ifm_index).get();

  auto fn = std::make_unique<ops::MaxPoolLayer>();

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

  auto ofm_alloc = _tensor_builder->portableAt(ofm_index).get();
  auto ifm_alloc = _tensor_builder->portableAt(ifm_index).get();

  auto fn = std::make_unique<ops::AvgPoolLayer>();

  fn->configure(ifm_alloc, padding.left, padding.right, padding.top, padding.bottom,
                stride.horizontal, stride.vertical, kw, kh, activation, ofm_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Concat &node)
{
  const auto ofm_index{node.getOutputs().at(0)};

  const auto rank = _ctx.at(ofm_index).shape().rank();
  const auto axis = ops::getAxis(rank, node.param().axis, _current_op_seq_layout);

  auto output_alloc = _tensor_builder->portableAt(ofm_index).get();

  std::vector<const IPortableTensor *> input_tensors;
  for (auto &ifm_idx : node.getInputs())
    input_tensors.emplace_back(_tensor_builder->portableAt(ifm_idx).get());

  auto fn = std::make_unique<ops::ConcatLayer>();

  fn->configure(input_tensors, axis, output_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Fill &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Fill::Input::INPUT)};
  const auto value_index{node.getInputs().at(ir::operation::Fill::Input::VALUE)};

  auto output_alloc = _tensor_builder->portableAt(output_index).get();
  auto input_alloc = _tensor_builder->portableAt(input_index).get();
  auto value_alloc = _tensor_builder->portableAt(value_index).get();

  auto fn = std::make_unique<ops::FillLayer>();

  fn->configure(input_alloc, value_alloc, output_alloc);

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

  auto output_alloc = _tensor_builder->portableAt(output_index).get();
  auto input_alloc = _tensor_builder->portableAt(input_index).get();
  auto weight_alloc = _tensor_builder->portableAt(weight_index).get();
  auto bias_alloc =
      bias_index.undefined() ? nullptr : _tensor_builder->portableAt(bias_index).get();

  auto fn = std::make_unique<ops::FullyConnectedLayer>();

  fn->configure(input_alloc, weight_alloc, bias_alloc, activation, output_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Reshape &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Reshape::Input::INPUT)};

  auto output_alloc = _tensor_builder->portableAt(output_index).get();
  auto input_alloc = _tensor_builder->portableAt(input_index).get();

  // optional 2nd input
  IPortableTensor *shape_alloc = nullptr;

  if (node.getInputs().size() == 2)
  {
    const auto shape_index{node.getInputs().at(ir::operation::Reshape::Input::SHAPE)};
    shape_alloc = _tensor_builder->portableAt(shape_index).get();
  }

  auto fn = std::make_unique<ops::ReshapeLayer>();

  fn->configure(input_alloc, shape_alloc, output_alloc);
  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Squeeze &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Squeeze::Input::INPUT)};

  auto output_alloc = _tensor_builder->portableAt(output_index).get();
  auto input_alloc = _tensor_builder->portableAt(input_index).get();

  // Squeeze can share same kernel with reshape
  auto fn = std::make_unique<ops::ReshapeLayer>();

  fn->configure(input_alloc, nullptr, output_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Softmax &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Softmax::Input::INPUT)};

  const auto beta = node.param().beta;

  auto output_alloc = _tensor_builder->portableAt(output_index).get();
  auto input_alloc = _tensor_builder->portableAt(input_index).get();

  auto fn = std::make_unique<ops::SoftMaxLayer>();

  fn->configure(input_alloc, beta, output_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Add &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto lhs_index{node.getInputs().at(ir::operation::Add::Input::LHS)};
  const auto rhs_index{node.getInputs().at(ir::operation::Add::Input::RHS)};

  const auto activation = node.param().activation;

  auto ofm_alloc = _tensor_builder->portableAt(ofm_index).get();
  auto lhs_alloc = _tensor_builder->portableAt(lhs_index).get();
  auto rhs_alloc = _tensor_builder->portableAt(rhs_index).get();

  auto fn = std::make_unique<ops::AddLayer>();

  fn->configure(lhs_alloc, rhs_alloc, activation, ofm_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Comparison &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto lhs_index{node.getInputs().at(ir::operation::Comparison::Input::INPUT0)};
  const auto rhs_index{node.getInputs().at(ir::operation::Comparison::Input::INPUT1)};

  auto ofm_alloc = _tensor_builder->portableAt(ofm_index).get();
  auto lhs_alloc = _tensor_builder->portableAt(lhs_index).get();
  auto rhs_alloc = _tensor_builder->portableAt(rhs_index).get();

  auto comparison_type = node.param().comparison_type;

  auto fn = std::make_unique<ops::CompareLayer>();

  fn->configure(lhs_alloc, rhs_alloc, comparison_type, ofm_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Gather &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Gather::Input::INPUT)};
  const auto indices_index{node.getInputs().at(ir::operation::Gather::Input::INDICES)};

  auto output_alloc = _tensor_builder->portableAt(output_index).get();
  auto input_alloc = _tensor_builder->portableAt(input_index).get();
  auto indices_alloc = _tensor_builder->portableAt(indices_index).get();

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

  auto fn = std::make_unique<ops::GatherLayer>();

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

  auto ofm_alloc = _tensor_builder->portableAt(ofm_index).get();
  auto lhs_alloc = _tensor_builder->portableAt(lhs_index).get();
  auto rhs_alloc = _tensor_builder->portableAt(rhs_index).get();

  auto fn = std::make_unique<ops::SubLayer>();

  fn->configure(lhs_alloc, rhs_alloc, activation, ofm_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Mul &node)
{
  // The same as Add
  const auto ofm_index{node.getOutputs().at(0)};
  const auto lhs_index{node.getInputs().at(ir::operation::Mul::Input::LHS)};
  const auto rhs_index{node.getInputs().at(ir::operation::Mul::Input::RHS)};

  const auto activation = node.param().activation;

  auto ofm_alloc = _tensor_builder->portableAt(ofm_index).get();
  auto lhs_alloc = _tensor_builder->portableAt(lhs_index).get();
  auto rhs_alloc = _tensor_builder->portableAt(rhs_index).get();

  auto fn = std::make_unique<ops::MulLayer>();

  fn->configure(lhs_alloc, rhs_alloc, activation, ofm_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::OneHot &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto indices_index{node.getInputs().at(ir::operation::OneHot::INDICES)};
  const auto depth_index{node.getInputs().at(ir::operation::OneHot::Input::DEPTH)};
  const auto onvalue_index{node.getInputs().at(ir::operation::OneHot::Input::ON_VALUE)};
  const auto offvalue_index{node.getInputs().at(ir::operation::OneHot::Input::OFF_VALUE)};

  const auto axis = node.param().axis;

  auto output_alloc = _tensor_builder->portableAt(output_index).get();
  auto indices_alloc = _tensor_builder->portableAt(indices_index).get();
  auto depth_alloc = _tensor_builder->portableAt(depth_index).get();
  auto onvalue_alloc = _tensor_builder->portableAt(onvalue_index).get();
  auto offvalue_alloc = _tensor_builder->portableAt(offvalue_index).get();

  assert(indices_alloc->data_type() == OperandType::INT32);
  assert(axis <= static_cast<int>(indices_alloc->num_dimensions()));

  auto fn = std::make_unique<ops::OneHotLayer>();

  fn->configure(indices_alloc, depth_alloc, onvalue_alloc, offvalue_alloc, output_alloc, axis);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Div &node)
{
  // The same as Add
  const auto ofm_index{node.getOutputs().at(0)};
  const auto lhs_index{node.getInputs().at(ir::operation::Div::Input::LHS)};
  const auto rhs_index{node.getInputs().at(ir::operation::Div::Input::RHS)};

  const auto activation = node.param().activation;

  auto ofm_alloc = _tensor_builder->portableAt(ofm_index).get();
  auto lhs_alloc = _tensor_builder->portableAt(lhs_index).get();
  auto rhs_alloc = _tensor_builder->portableAt(rhs_index).get();

  auto fn = std::make_unique<ops::DivLayer>();

  fn->configure(lhs_alloc, rhs_alloc, activation, ofm_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Einsum &node)
{
  const auto ofm_index{node.getOutputs().at(0)};

  auto output_alloc = _tensor_builder->portableAt(ofm_index).get();
  std::vector<const IPortableTensor *> input_allocs;
  for (auto &ifm_idx : node.getInputs())
    input_allocs.emplace_back(_tensor_builder->portableAt(ifm_idx).get());

  const auto equation = node.param().equation;

  auto fn = std::make_unique<ops::EinsumLayer>();

  fn->configure(input_allocs, equation, output_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Custom &node)
{
  auto get_type_info = [](const ir::Operand &operand) -> custom::TypeInfo {
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
      auto in_alloc = _tensor_builder->portableAt(idx)->buffer();
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

  auto output_alloc = _tensor_builder->portableAt(output_index).get();
  auto input_alloc = _tensor_builder->portableAt(input_index).get();

  auto fn = std::make_unique<ops::ExpLayer>();

  fn->configure(input_alloc, output_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::ExpandDims &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::ExpandDims::Input::INPUT)};
  const auto axis_index{node.getInputs().at(ir::operation::ExpandDims::Input::AXIS)};

  auto output_alloc = _tensor_builder->portableAt(output_index).get();
  auto input_alloc = _tensor_builder->portableAt(input_index).get();
  auto axis_alloc = _tensor_builder->portableAt(axis_index).get();

  auto fn = std::make_unique<ops::ExpandDimsLayer>();

  fn->configure(input_alloc, axis_alloc, output_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Logistic &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Logistic::Input::INPUT)};

  auto output_alloc = _tensor_builder->portableAt(output_index).get();
  auto input_alloc = _tensor_builder->portableAt(input_index).get();

  auto fn = std::make_unique<ops::LogisticLayer>();

  fn->configure(input_alloc, output_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Tanh &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Tanh::Input::INPUT)};

  auto output_alloc = _tensor_builder->portableAt(output_index).get();
  auto input_alloc = _tensor_builder->portableAt(input_index).get();

  auto fn = std::make_unique<ops::TanhLayer>();

  fn->configure(input_alloc, output_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Pack &node)
{
  const auto ofm_index{node.getOutputs().at(0)};

  const auto rank = _ctx.at(ofm_index).shape().rank();
  const auto axis = ops::getAxis(rank, node.param().axis, _current_op_seq_layout);

  assert(-rank <= axis && axis < rank);

  auto output_alloc = _tensor_builder->portableAt(ofm_index).get();

  std::vector<const IPortableTensor *> input_tensors;
  for (auto &ifm_idx : node.getInputs())
    input_tensors.emplace_back(_tensor_builder->portableAt(ifm_idx).get());

  auto fn = std::make_unique<ops::PackLayer>();

  fn->configure(input_tensors, axis, output_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Unpack &node)
{
  const auto input_index{node.getInputs().at(0)};

  const auto rank = _ctx.at(input_index).shape().rank();
  const auto axis = ops::getAxis(rank, node.param().axis, _current_op_seq_layout);

  assert(rank == 0 || (-rank <= axis && axis < rank));

  auto input_alloc = _tensor_builder->portableAt(input_index).get();

  std::vector<IPortableTensor *> output_tensors;
  for (auto &output_idx : node.getOutputs())
    output_tensors.emplace_back(_tensor_builder->portableAt(output_idx).get());

  auto fn = std::make_unique<ops::UnpackLayer>();

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

  auto input = _tensor_builder->portableAt(input_index).get();
  auto output = _tensor_builder->portableAt(output_index).get();
  auto pad_rank = _ctx.at(pad_index).shape().dim(0);
  auto pad_base = reinterpret_cast<const int32_t *>(_ctx.at(pad_index).data()->base());

  auto fn = std::make_unique<ops::PadLayer>();

  fn->configure(input, output, pad_base, pad_rank);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Max &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto lhs_index{node.getInputs().at(ir::operation::Max::Input::LHS)};
  const auto rhs_index{node.getInputs().at(ir::operation::Max::Input::RHS)};

  auto ofm_alloc = _tensor_builder->portableAt(ofm_index).get();
  auto lhs_alloc = _tensor_builder->portableAt(lhs_index).get();
  auto rhs_alloc = _tensor_builder->portableAt(rhs_index).get();

  auto fn = std::make_unique<ops::MaxLayer>();

  fn->configure(lhs_alloc, rhs_alloc, ofm_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Min &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto lhs_index{node.getInputs().at(ir::operation::Min::Input::LHS)};
  const auto rhs_index{node.getInputs().at(ir::operation::Min::Input::RHS)};

  auto ofm_alloc = _tensor_builder->portableAt(ofm_index).get();
  auto lhs_alloc = _tensor_builder->portableAt(lhs_index).get();
  auto rhs_alloc = _tensor_builder->portableAt(rhs_index).get();

  auto fn = std::make_unique<ops::MinLayer>();

  fn->configure(lhs_alloc, rhs_alloc, ofm_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Cast &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::Cast::Input::INPUT)};

  auto ofm_alloc = _tensor_builder->portableAt(ofm_index).get();
  auto ifm_alloc = _tensor_builder->portableAt(ifm_index).get();

  auto fn = std::make_unique<ops::CastLayer>();

  fn->configure(ifm_alloc, ofm_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Transpose &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Transpose::Input::INPUT)};

  auto output_alloc = _tensor_builder->portableAt(output_index).get();
  auto input_alloc = _tensor_builder->portableAt(input_index).get();

  auto fn = std::make_unique<ops::TransposeLayer>();

  fn->configure(input_alloc, output_alloc, node.param().perm);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::ReduceSum &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::ReduceSum::Input::INPUT)};
  const auto axes_index{node.getInputs().at(ir::operation::ReduceSum::Input::AXES)};

  const auto keep_dims = node.param().keep_dims;
  auto output_alloc = _tensor_builder->portableAt(output_index).get();
  auto input_alloc = _tensor_builder->portableAt(input_index).get();
  auto axes_alloc = _tensor_builder->portableAt(axes_index).get();

  auto fn = std::make_unique<ops::ReduceLayer>();

  fn->configure(input_alloc, axes_alloc, output_alloc, ops::ReduceType::kSum, keep_dims);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::ReduceAll &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::ReduceAll::Input::INPUT)};
  const auto axes_index{node.getInputs().at(ir::operation::ReduceAll::Input::AXES)};

  const auto keep_dims = node.param().keep_dims;
  auto output_alloc = _tensor_builder->portableAt(output_index).get();
  auto input_alloc = _tensor_builder->portableAt(input_index).get();
  auto axes_alloc = _tensor_builder->portableAt(axes_index).get();

  auto fn = std::make_unique<ops::ReduceLayer>();

  fn->configure(input_alloc, axes_alloc, output_alloc, ops::ReduceType::kAll, keep_dims);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::ReduceAny &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::ReduceAny::Input::INPUT)};
  const auto axes_index{node.getInputs().at(ir::operation::ReduceAny::Input::AXES)};

  const auto keep_dims = node.param().keep_dims;
  auto output_alloc = _tensor_builder->portableAt(output_index).get();
  auto input_alloc = _tensor_builder->portableAt(input_index).get();
  auto axes_alloc = _tensor_builder->portableAt(axes_index).get();

  auto fn = std::make_unique<ops::ReduceLayer>();

  fn->configure(input_alloc, axes_alloc, output_alloc, ops::ReduceType::kAny, keep_dims);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::ReduceMax &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::ReduceMax::INPUT)};
  const auto axes_index{node.getInputs().at(ir::operation::ReduceMax::AXES)};

  const auto keep_dims = node.param().keep_dims;
  auto output_alloc = _tensor_builder->portableAt(output_index).get();
  auto input_alloc = _tensor_builder->portableAt(input_index).get();
  auto axes_alloc = _tensor_builder->portableAt(axes_index).get();

  auto fn = std::make_unique<ops::ReduceLayer>();

  fn->configure(input_alloc, axes_alloc, output_alloc, ops::ReduceType::kMax, keep_dims);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::ReduceMin &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::ReduceMin::Input::INPUT)};
  const auto axes_index{node.getInputs().at(ir::operation::ReduceMin::Input::AXES)};

  const auto keep_dims = node.param().keep_dims;
  auto output_alloc = _tensor_builder->portableAt(output_index).get();
  auto input_alloc = _tensor_builder->portableAt(input_index).get();
  auto axes_alloc = _tensor_builder->portableAt(axes_index).get();

  auto fn = std::make_unique<ops::ReduceLayer>();

  fn->configure(input_alloc, axes_alloc, output_alloc, ops::ReduceType::kMin, keep_dims);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::ReLU &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(0)};

  auto output_alloc = _tensor_builder->portableAt(output_index).get();
  auto input_alloc = _tensor_builder->portableAt(input_index).get();

  auto fn = std::make_unique<ops::ReLULayer>();

  fn->configure(input_alloc, output_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Select &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto condition_index{node.getInputs().at(ir::operation::Select::Input::CONDITION)};
  const auto true_index{node.getInputs().at(ir::operation::Select::Input::INPUT_TRUE)};
  const auto false_index{node.getInputs().at(ir::operation::Select::Input::INPUT_FALSE)};

  auto output_alloc = _tensor_builder->portableAt(output_index).get();
  auto condition_alloc = _tensor_builder->portableAt(condition_index).get();
  auto true_alloc = _tensor_builder->portableAt(true_index).get();
  auto false_alloc = _tensor_builder->portableAt(false_index).get();

  auto fn = std::make_unique<ops::SelectLayer>();

  fn->configure(condition_alloc, true_alloc, false_alloc, output_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Slice &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Slice::Input::INPUT)};
  const auto begins_index{node.getInputs().at(ir::operation::Slice::Input::BEGINS)};
  const auto sizes_index{node.getInputs().at(ir::operation::Slice::Input::SIZES)};

  auto output_alloc = _tensor_builder->portableAt(output_index).get();
  auto input_alloc = _tensor_builder->portableAt(input_index).get();
  auto begins_alloc = _tensor_builder->portableAt(begins_index).get();
  auto sizes_alloc = _tensor_builder->portableAt(sizes_index).get();

  auto fn = std::make_unique<ops::SliceLayer>();

  fn->configure(input_alloc, begins_alloc, sizes_alloc, output_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::StridedSlice &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::StridedSlice::Input::INPUT)};
  const auto starts_index{node.getInputs().at(ir::operation::StridedSlice::Input::STARTS)};
  const auto ends_index{node.getInputs().at(ir::operation::StridedSlice::Input::ENDS)};
  const auto strides_index{node.getInputs().at(ir::operation::StridedSlice::Input::STRIDES)};

  auto output_alloc = _tensor_builder->portableAt(output_index).get();
  auto input_alloc = _tensor_builder->portableAt(input_index).get();
  auto starts_alloc = _tensor_builder->portableAt(starts_index).get();
  auto ends_alloc = _tensor_builder->portableAt(ends_index).get();
  auto strides_alloc = _tensor_builder->portableAt(strides_index).get();

  auto begin_mask = node.param().begin_mask;
  auto end_mask = node.param().end_mask;
  auto shrink_axis_mask = node.param().shrink_axis_mask;

  auto fn = std::make_unique<ops::StridedSliceLayer>();

  fn->configure(input_alloc, starts_alloc, ends_alloc, strides_alloc, output_alloc, begin_mask,
                end_mask, shrink_axis_mask);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Split &node)
{
  const auto num_splits = node.param().num_splits;
  assert(num_splits == static_cast<int>(node.getOutputs().size()));

  const auto input_idx{node.getInputs().at(ir::operation::Split::Input::INPUT)};
  const auto rank = _ctx.at(input_idx).shape().rank();
  const auto axis = ops::getAxis(rank, node.param().axis, _current_op_seq_layout);
  auto axis_resolved = axis < 0 ? axis + rank : axis;

  auto in_tensor = _tensor_builder->portableAt(input_idx).get();

  std::vector<IPortableTensor *> out_tensors;
  for (auto &output_idx : node.getOutputs())
    out_tensors.emplace_back(_tensor_builder->portableAt(output_idx).get());

  auto fn = std::make_unique<ops::SplitLayer>();

  fn->configure(in_tensor, num_splits, axis_resolved, out_tensors);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Abs &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::Abs::Input::INPUT)};

  auto ofm_alloc = _tensor_builder->portableAt(ofm_index).get();
  auto ifm_alloc = _tensor_builder->portableAt(ifm_index).get();

  auto fn = std::make_unique<ops::AbsLayer>();

  fn->configure(ifm_alloc, ofm_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Sin &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::Sin::Input::INPUT)};

  auto ofm_alloc = _tensor_builder->portableAt(ofm_index).get();
  auto ifm_alloc = _tensor_builder->portableAt(ifm_index).get();

  auto fn = std::make_unique<ops::SinLayer>();

  fn->configure(ifm_alloc, ofm_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Cos &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::Cos::Input::INPUT)};

  auto ofm_alloc = _tensor_builder->portableAt(ofm_index).get();
  auto ifm_alloc = _tensor_builder->portableAt(ifm_index).get();

  auto fn = std::make_unique<ops::CosLayer>();

  fn->configure(ifm_alloc, ofm_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::RSQRT &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::RSQRT::Input::INPUT)};

  auto ofm_alloc = _tensor_builder->portableAt(ofm_index).get();
  auto ifm_alloc = _tensor_builder->portableAt(ifm_index).get();

  auto fn = std::make_unique<ops::RsqrtLayer>();

  fn->configure(ifm_alloc, ofm_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Shape &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::Shape::Input::INPUT)};

  auto ofm_alloc = _tensor_builder->portableAt(ofm_index).get();
  auto ifm_alloc = _tensor_builder->portableAt(ifm_index).get();

  auto fn = std::make_unique<ops::ShapeLayer>();

  fn->configure(ifm_alloc, ofm_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::ReduceProd &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::ReduceProd::Input::INPUT)};
  const auto axes_index{node.getInputs().at(ir::operation::ReduceProd::Input::AXES)};

  const auto keep_dims = node.param().keep_dims;
  auto output_alloc = _tensor_builder->portableAt(output_index).get();
  auto input_alloc = _tensor_builder->portableAt(input_index).get();
  auto axes_alloc = _tensor_builder->portableAt(axes_index).get();

  auto fn = std::make_unique<ops::ReduceLayer>();

  fn->configure(input_alloc, axes_alloc, output_alloc, ops::ReduceType::kProd, keep_dims);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Reverse &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Reverse::INPUT)};
  const auto axis_index{node.getInputs().at(ir::operation::Reverse::AXIS)};

  auto output_alloc = _tensor_builder->portableAt(output_index).get();
  auto input_alloc = _tensor_builder->portableAt(input_index).get();
  auto axis_alloc = _tensor_builder->portableAt(axis_index).get();

  auto fn = std::make_unique<ops::ReverseLayer>();

  fn->configure(input_alloc, axis_alloc, output_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Neg &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::Neg::Input::INPUT)};

  auto ofm_alloc = _tensor_builder->portableAt(ofm_index).get();
  auto ifm_alloc = _tensor_builder->portableAt(ifm_index).get();

  auto fn = std::make_unique<ops::NegLayer>();

  fn->configure(ifm_alloc, ofm_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::ArgMax &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::ArgMax::INPUT)};

  const auto axis = node.param().axis;

  auto output_alloc = _tensor_builder->portableAt(output_index).get();
  auto input_alloc = _tensor_builder->portableAt(input_index).get();

  auto fn = std::make_unique<ops::ArgMinMaxLayer>();

  fn->configure(input_alloc, output_alloc, axis, /* is_arg_max */ true);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Pow &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto lhs_index{node.getInputs().at(ir::operation::Pow::LHS)};
  const auto rhs_index{node.getInputs().at(ir::operation::Pow::RHS)};

  auto output_alloc = _tensor_builder->portableAt(output_index).get();
  auto lhs_alloc = _tensor_builder->portableAt(lhs_index).get();
  auto rhs_alloc = _tensor_builder->portableAt(rhs_index).get();

  auto fn = std::make_unique<ops::PowLayer>();

  fn->configure(lhs_alloc, rhs_alloc, ir::Activation::NONE, output_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Log &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::Log::Input::INPUT)};

  auto ofm_alloc = _tensor_builder->portableAt(ofm_index).get();
  auto ifm_alloc = _tensor_builder->portableAt(ifm_index).get();

  auto fn = std::make_unique<ops::LogLayer>();

  fn->configure(ifm_alloc, ofm_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Round &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Round::INPUT)};

  auto output_alloc = _tensor_builder->portableAt(output_index).get();
  auto input_alloc = _tensor_builder->portableAt(input_index).get();

  auto fn = std::make_unique<ops::RoundLayer>();

  fn->configure(input_alloc, output_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::LogicalNot &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::LogicalNot::INPUT)};

  auto output_alloc = _tensor_builder->portableAt(output_index).get();
  auto input_alloc = _tensor_builder->portableAt(input_index).get();

  auto fn = std::make_unique<ops::LogicalNotLayer>();

  fn->configure(input_alloc, output_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::LogicalOr &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto lhs_index{node.getInputs().at(0)};
  const auto rhs_index{node.getInputs().at(1)};

  auto ofm_alloc = _tensor_builder->portableAt(ofm_index).get();
  auto lhs_alloc = _tensor_builder->portableAt(lhs_index).get();
  auto rhs_alloc = _tensor_builder->portableAt(rhs_index).get();

  auto fn = std::make_unique<ops::LogicalOrLayer>();

  fn->configure(lhs_alloc, rhs_alloc, ofm_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Mean &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Mean::INPUT)};
  const auto axes_index{node.getInputs().at(ir::operation::Mean::AXES)};

  const auto keep_dims = node.param().keep_dims;
  auto output_alloc = _tensor_builder->portableAt(output_index).get();
  auto input_alloc = _tensor_builder->portableAt(input_index).get();
  auto axes_alloc = _tensor_builder->portableAt(axes_index).get();

  auto fn = std::make_unique<ops::MeanLayer>();

  fn->configure(input_alloc, axes_alloc, output_alloc, keep_dims);
  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::ZerosLike &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::ZerosLike::INPUT)};

  auto output_alloc = _tensor_builder->portableAt(output_index).get();
  auto input_alloc = _tensor_builder->portableAt(input_index).get();

  auto fn = std::make_unique<ops::ZerosLikeLayer>();

  fn->configure(input_alloc, output_alloc);
  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Range &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto start_index{node.getInputs().at(ir::operation::Range::START)};
  const auto limit_index{node.getInputs().at(ir::operation::Range::LIMIT)};
  const auto delta_index{node.getInputs().at(ir::operation::Range::DELTA)};

  auto output_alloc = _tensor_builder->portableAt(output_index).get();
  auto start_alloc = _tensor_builder->portableAt(start_index).get();
  auto limit_alloc = _tensor_builder->portableAt(limit_index).get();
  auto delta_alloc = _tensor_builder->portableAt(delta_index).get();

  auto fn = std::make_unique<ops::RangeLayer>();

  fn->configure(start_alloc, limit_alloc, delta_alloc, output_alloc);
  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::SquaredDifference &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto lhs_index{node.getInputs().at(ir::operation::SquaredDifference::Input::LHS)};
  const auto rhs_index{node.getInputs().at(ir::operation::SquaredDifference::Input::RHS)};

  auto ofm_alloc = _tensor_builder->portableAt(ofm_index).get();
  auto lhs_alloc = _tensor_builder->portableAt(lhs_index).get();
  auto rhs_alloc = _tensor_builder->portableAt(rhs_index).get();

  auto fn = std::make_unique<ops::SqDiffLayer>();

  fn->configure(lhs_alloc, rhs_alloc, ofm_alloc);
  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Tile &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Tile::INPUT)};
  const auto multiples_index{node.getInputs().at(ir::operation::Tile::MULTIPLES)};

  auto output_alloc = _tensor_builder->portableAt(output_index).get();
  auto input_alloc = _tensor_builder->portableAt(input_index).get();
  auto multiples_alloc = _tensor_builder->portableAt(multiples_index).get();

  auto fn = std::make_unique<ops::TileLayer>();

  fn->configure(input_alloc, multiples_alloc, output_alloc);
  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::MatrixBandPart &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::MatrixBandPart::INPUT)};
  const auto num_lower_index{node.getInputs().at(ir::operation::MatrixBandPart::NUM_LOWER_DIAG)};
  const auto num_upper_index{node.getInputs().at(ir::operation::MatrixBandPart::NUM_UPPER_DIAG)};

  auto output_alloc = _tensor_builder->portableAt(output_index).get();
  auto input_alloc = _tensor_builder->portableAt(input_index).get();
  auto num_lower_alloc = _tensor_builder->portableAt(num_lower_index).get();
  auto num_upper_alloc = _tensor_builder->portableAt(num_upper_index).get();

  auto fn = std::make_unique<ops::MatrixBandPartLayer>();

  fn->configure(input_alloc, num_lower_alloc, num_upper_alloc, output_alloc);
  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::BatchMatMul &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto lhs_index{node.getInputs().at(ir::operation::BatchMatMul::LHS)};
  const auto rhs_index{node.getInputs().at(ir::operation::BatchMatMul::RHS)};

  auto output_alloc = _tensor_builder->portableAt(output_index).get();
  auto lhs_alloc = _tensor_builder->portableAt(lhs_index).get();
  auto rhs_alloc = _tensor_builder->portableAt(rhs_index).get();

  const auto adj_x = node.param().adj_x;
  const auto adj_y = node.param().adj_y;

  auto fn = std::make_unique<ops::BatchMatMulLayer>();

  fn->configure(lhs_alloc, rhs_alloc, adj_x, adj_y, output_alloc);
  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::BroadcastTo &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::BroadcastTo::INPUT)};
  const auto shape_index{node.getInputs().at(ir::operation::BroadcastTo::SHAPE)};

  auto output_alloc = _tensor_builder->at(output_index).get();
  auto input_alloc = _tensor_builder->at(input_index).get();
  auto shape_alloc = _tensor_builder->at(shape_index).get();

  auto fn = std::make_unique<ops::BroadcastToLayer>();

  fn->configure(input_alloc, shape_alloc, output_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::FusedBatchNorm &node)
{
  const auto ofm_index{node.getOutputs().at(0)};

  auto output_alloc = _tensor_builder->at(ofm_index).get();
  std::vector<const Tensor *> input_allocs;
  for (auto &ifm_idx : node.getInputs())
    input_allocs.emplace_back(_tensor_builder->at(ifm_idx).get());

  const auto epsilon = node.param().epsilon;
  const auto is_training = node.param().is_training;
  const auto data_format = node.param().data_format;

  auto fn = std::make_unique<ops::FusedBatchNormLayer>();

  fn->configure(input_allocs, epsilon, is_training, data_format, output_alloc);

  _return_fn = std::move(fn);
}

} // namespace cpu
} // namespace backend
} // namespace onert
