/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <arm_compute/runtime/CL/CLFunctions.h>   // Include all ARM Compute CL functions
#include <arm_compute/runtime/CL/CLFunctionsEx.h> // Include all ARM Compute EX CL functions

#include <AclActivationBuilder.h>
#include <AclFunction.h>
#include <Convert.h>
#include <Swizzle.h>

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
namespace acl_cl
{

using ::onert::backend::acl_common::asAclClFunction;
using ActivationBuilder = ::onert::backend::acl_common::AclActivationBuilder<
    ::arm_compute::ICLTensor, ::arm_compute::CLActivationLayer, acl_common::AclClFunction>;

KernelGenerator::KernelGenerator(const ir::Operands &operands_ctx,
                                 const ir::Operations &operations_ctx,
                                 const std::shared_ptr<TensorBuilder> &tensor_builder)
    : _ctx(operands_ctx), _operations_ctx(operations_ctx), _tensor_builder(tensor_builder),
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
  for (const auto &operation_idx : op_seq.operations())
  {
    const auto &node = _operations_ctx.at(operation_idx);
    node.accept(*this);
    _return_fn_seq->append(releaseFunction());
  }
}

void KernelGenerator::visit(const ir::operation::BatchToSpaceND &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::BatchToSpaceND::Input::INPUT)};
  const auto block_size_index{
      node.getInputs().at(ir::operation::BatchToSpaceND::Input::BLOCK_SIZE)};

  auto ofm_alloc = _tensor_builder->at(ofm_index).get();
  auto ifm_alloc = _tensor_builder->at(ifm_index).get();
  auto block_size_alloc = _tensor_builder->at(block_size_index).get();

  assert(_ctx.at(block_size_index).data());

  auto fn = std::make_unique<::arm_compute::CLBatchToSpaceLayer>();

  fn->configure(ifm_alloc->handle(), block_size_alloc->handle(), ofm_alloc->handle());

  auto acl_fn = asAclClFunction(std::move(fn));

  _return_fn = std::move(acl_fn);
}

void KernelGenerator::visit(const ir::operation::Cast &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::Cast::Input::INPUT)};

  auto ofm_alloc = _tensor_builder->at(ofm_index).get();
  auto ifm_alloc = _tensor_builder->at(ifm_index).get();
  const auto input_sub_type = _ctx.at(ifm_index).typeInfo().type() == ir::DataType::BOOL8
                                  ? arm_compute::SubDataType::BOOL
                                  : arm_compute::SubDataType::NONE;

  auto fn = std::make_unique<::arm_compute::CLCast>();

  fn->configure(ifm_alloc->handle(), ofm_alloc->handle(), input_sub_type);

  auto acl_fn = asAclClFunction(std::move(fn));

  _return_fn = std::move(acl_fn);
}

void KernelGenerator::visit(const ir::operation::Conv2D &node)
{
  using ir::operation::Conv2D;

  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(Conv2D::Input::INPUT)};
  const auto ker_index{node.getInputs().at(Conv2D::Input::KERNEL)};
  const auto bias_index{node.getInputs().at(Conv2D::Input::BIAS)};

  const auto ifm_shape = _ctx.at(ifm_index).shape().asFeature(_current_op_seq_layout);
  const auto ofm_shape = _ctx.at(ofm_index).shape().asFeature(_current_op_seq_layout);
  // Kernel format is [depth_out, kernel_height, kernel_width, depth_in].
  const auto &ker_shape = _ctx.at(ker_index).shape();
  const auto ker_height = ker_shape.dim(1);
  const auto ker_width = ker_shape.dim(2);

  const auto stride = node.param().stride;
  const auto padding = ir::calculatePadding(node.param().padding, ifm_shape, ofm_shape, stride,
                                            ker_width, ker_height);
  const auto activation = node.param().activation;

  auto ofm_alloc = _tensor_builder->at(ofm_index).get();
  auto ifm_alloc = _tensor_builder->at(ifm_index).get();
  auto ker_alloc = _tensor_builder->at(ker_index).get();
  auto bias_alloc = _tensor_builder->at(bias_index).get();

  const auto conv_info = acl_common::asPadStrideInfo(padding, stride);
  const auto act_info = acl_common::asActivationLayerInfo(activation);

  auto fn = std::make_unique<::arm_compute::CLConvolutionLayer>(
      _tensor_builder->acl_tensor_manager()->internal_buffer_manager());

  fn->configure(ifm_alloc->handle(), ker_alloc->handle(), bias_alloc->handle(), ofm_alloc->handle(),
                conv_info, ::arm_compute::WeightsInfo(), ::arm_compute::Size2D(1U, 1U), act_info);

  _return_fn = asAclClFunction(std::move(fn));
}

void KernelGenerator::visit(const ir::operation::DepthwiseConv2D &node)
{
  using ir::operation::DepthwiseConv2D;

  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(DepthwiseConv2D::Input::INPUT)};
  const auto ker_index{node.getInputs().at(DepthwiseConv2D::Input::KERNEL)};
  const auto bias_index{node.getInputs().at(DepthwiseConv2D::Input::BIAS)};

  const auto ifm_shape = _ctx.at(ifm_index).shape().asFeature(_current_op_seq_layout);
  const auto ofm_shape = _ctx.at(ofm_index).shape().asFeature(_current_op_seq_layout);
  // Kernel format is [1, kernel_height, kernel_width, depth_out].
  const auto &ker_shape = _ctx.at(ker_index).shape();
  const auto ker_height = ker_shape.dim(1);
  const auto ker_width = ker_shape.dim(2);

  const auto stride = node.param().stride;
  const auto padding = ir::calculatePadding(node.param().padding, ifm_shape, ofm_shape, stride,
                                            ker_width, ker_height);
  const auto multiplier = node.param().multiplier;
  const auto activation = node.param().activation;

  auto ofm_alloc = _tensor_builder->at(ofm_index).get();
  auto ifm_alloc = _tensor_builder->at(ifm_index).get();
  auto ker_alloc = _tensor_builder->at(ker_index).get();
  auto bias_alloc = _tensor_builder->at(bias_index).get();

  const auto conv_info = acl_common::asPadStrideInfo(padding, stride);
  const auto act_info = acl_common::asActivationLayerInfo(activation);

  {
    auto fn = std::make_unique<::arm_compute::CLDepthwiseConvolutionLayer>();

    fn->configure(ifm_alloc->handle(), ker_alloc->handle(), bias_alloc->handle(),
                  ofm_alloc->handle(), conv_info, multiplier, act_info);

    _return_fn = asAclClFunction(std::move(fn));
  }
}

void KernelGenerator::visit(const ir::operation::MaxPool2D &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::MaxPool2D::Input::INPUT)};

  const auto ofm_shape = _ctx.at(ofm_index).shape().asFeature(_current_op_seq_layout);
  const auto ifm_shape = _ctx.at(ifm_index).shape().asFeature(_current_op_seq_layout);

  const auto kh = node.param().kh;
  const auto kw = node.param().kw;
  const auto stride = node.param().stride;
  const auto padding =
      ir::calculatePadding(node.param().padding, ifm_shape, ofm_shape, stride, kw, kh);
  const auto activation = node.param().activation;

  VERBOSE(MaxPool2D) << "IFM_H: " << ifm_shape.H << std::endl;
  VERBOSE(MaxPool2D) << "IFM_W: " << ifm_shape.W << std::endl;
  VERBOSE(MaxPool2D) << "OFM_H: " << ofm_shape.H << std::endl;
  VERBOSE(MaxPool2D) << "OFM_W: " << ofm_shape.W << std::endl;
  VERBOSE(MaxPool2D) << "KER_H: " << kh << std::endl;
  VERBOSE(MaxPool2D) << "KER_W: " << kw << std::endl;
  VERBOSE(MaxPool2D) << "STRIDE_H: " << stride.vertical << std::endl;
  VERBOSE(MaxPool2D) << "STRIDE_W: " << stride.horizontal << std::endl;
  VERBOSE(MaxPool2D) << "PAD(T): " << padding.top << std::endl;
  VERBOSE(MaxPool2D) << "PAD(B): " << padding.bottom << std::endl;
  VERBOSE(MaxPool2D) << "PAD(L): " << padding.left << std::endl;
  VERBOSE(MaxPool2D) << "PAD(R): " << padding.right << std::endl;

  auto ofm_alloc = _tensor_builder->at(ofm_index).get();
  auto ifm_alloc = _tensor_builder->at(ifm_index).get();

  ::arm_compute::PoolingLayerInfo info{::arm_compute::PoolingType::MAX,
                                       ::arm_compute::Size2D{kw, kh},
                                       acl_common::asPadStrideInfo(padding, stride)};

  auto fn = std::make_unique<::arm_compute::CLPoolingLayer>();

  fn->configure(ifm_alloc->handle(), ofm_alloc->handle(), info);

  _return_fn = std::make_unique<exec::FunctionSequence>(
      asAclClFunction(std::move(fn)), ActivationBuilder::generate(activation, ofm_alloc->handle()));
}

void KernelGenerator::visit(const ir::operation::AvgPool2D &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::AvgPool2D::Input::INPUT)};

  const auto ofm_shape = _ctx.at(ofm_index).shape().asFeature(_current_op_seq_layout);
  const auto ifm_shape = _ctx.at(ifm_index).shape().asFeature(_current_op_seq_layout);

  const auto kh = node.param().kh;
  const auto kw = node.param().kw;
  const auto stride = node.param().stride;
  const auto padding =
      ir::calculatePadding(node.param().padding, ifm_shape, ofm_shape, stride, kw, kh);
  const auto activation = node.param().activation;

  VERBOSE(AvgPool2D) << "IFM_H: " << ifm_shape.H << std::endl;
  VERBOSE(AvgPool2D) << "IFM_W: " << ifm_shape.W << std::endl;
  VERBOSE(AvgPool2D) << "OFM_H: " << ofm_shape.H << std::endl;
  VERBOSE(AvgPool2D) << "OFM_W: " << ofm_shape.W << std::endl;
  VERBOSE(AvgPool2D) << "KER_H: " << kh << std::endl;
  VERBOSE(AvgPool2D) << "KER_W: " << kw << std::endl;
  VERBOSE(AvgPool2D) << "STRIDE_H: " << stride.vertical << std::endl;
  VERBOSE(AvgPool2D) << "STRIDE_W: " << stride.horizontal << std::endl;
  VERBOSE(AvgPool2D) << "PAD(T): " << padding.top << std::endl;
  VERBOSE(AvgPool2D) << "PAD(B): " << padding.bottom << std::endl;
  VERBOSE(AvgPool2D) << "PAD(L): " << padding.left << std::endl;
  VERBOSE(AvgPool2D) << "PAD(R): " << padding.right << std::endl;

  auto ofm_alloc = _tensor_builder->at(ofm_index).get();
  auto ifm_alloc = _tensor_builder->at(ifm_index).get();

  ::arm_compute::PoolingLayerInfo info{
      ::arm_compute::PoolingType::AVG, ::arm_compute::Size2D{kw, kh},
      acl_common::asPadStrideInfo(padding, stride), true /* exclude_padding */};

  auto fn = std::make_unique<::arm_compute::CLPoolingLayer>();

  fn->configure(ifm_alloc->handle(), ofm_alloc->handle(), info);

  _return_fn = std::make_unique<exec::FunctionSequence>(
      asAclClFunction(std::move(fn)), ActivationBuilder::generate(activation, ofm_alloc->handle()));
}

void KernelGenerator::visit(const ir::operation::Concat &node)
{
  const auto ofm_index{node.getOutputs().at(0)};

  std::vector<ir::OperandIndex> input_indexes;

  for (const auto &input : node.getInputs())
    input_indexes.emplace_back(input);

  const auto axis = node.param().axis;

  // Concat elimination check
  bool eliminated = _tensor_builder->areSubTensorsOf(ofm_index, node.getInputs());
  if (eliminated)
  {
    // If concat eliminated, return a NOP IFunction
    VERBOSE(acl_cl_KernelGenerator_Concat) << "Concat eliminated" << std::endl;
    _return_fn = std::make_unique<exec::NopFunction>();
    return;
  }

  auto output_alloc = _tensor_builder->at(ofm_index).get();
  std::vector<::arm_compute::ICLTensor *> input_tensors;
  for (auto &ifm_ind : input_indexes)
    input_tensors.emplace_back(_tensor_builder->at(ifm_ind)->handle());

  std::unique_ptr<::arm_compute::IFunction> fn;
  if (input_indexes.size() < 2)
  {
    auto l = std::make_unique<::arm_compute::CLCopy>();
    l->configure(input_tensors.at(0), output_alloc->handle());
    fn = std::move(l);
  }
  else
  {
    auto l = std::make_unique<::arm_compute::CLConcatenateLayer>();
    const auto rank = _ctx.at(ofm_index).shape().rank();
    const auto frontend_layout = _current_op_seq_layout;
    const auto backend_layout = output_alloc->layout();
    const auto fixed_axis =
        acl_common::ToARMComputeAxis(rank, axis, frontend_layout, backend_layout).value();
    l->configure(input_tensors, output_alloc->handle(), fixed_axis);
    fn = std::move(l);
  }

  auto acl_fn = asAclClFunction(std::move(fn));

  _return_fn = std::move(acl_fn);
}

void KernelGenerator::visit(const ir::operation::FullyConnected &node)
{
  using ir::operation::FullyConnected;

  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(FullyConnected::Input::INPUT)};
  const auto weight_index{node.getInputs().at(FullyConnected::Input::WEIGHT)};
  const auto bias_index{node.getInputs().at(FullyConnected::Input::BIAS)};

  const auto input_rank = _ctx.at(input_index).shape().rank();

  const auto output_size =
      _ctx.at(output_index).shape().dim(_ctx.at(output_index).shape().rank() - 1);
  UNUSED_RELEASE(output_size);
  assert(_ctx.at(bias_index).shape().dim(0) == output_size);
  assert(_ctx.at(weight_index).shape().dim(0) == output_size);
  const auto batch_size =
      _ctx.at(output_index).shape().dim(_ctx.at(output_index).shape().rank() - 2);
  const auto input_size =
      _ctx.at(weight_index).shape().dim(_ctx.at(weight_index).shape().rank() - 1);

  // Check for reshaping input's shape into rank-2
  bool needs_reshape = false;
  ir::Shape reshape(2);
  if (input_rank == 3 || input_rank == 4)
  {
    const auto &ifm_shape = _ctx.at(input_index).shape();
    auto feature_size = 1;
    for (int i = 0; i < ifm_shape.rank(); ++i)
    {
      feature_size *= ifm_shape.dim(i);
    }

    UNUSED_RELEASE(feature_size);
    assert(feature_size == batch_size * input_size);

    // for reshaping
    needs_reshape = true;
    reshape.dim(0) = batch_size; /* H */
    reshape.dim(1) = input_size; /* W */
  }

  const auto activation = node.param().activation;

  auto output_alloc = _tensor_builder->at(output_index).get();
  const auto input_alloc = _tensor_builder->at(input_index).get();
  const auto weight_alloc = _tensor_builder->at(weight_index).get();
  const auto bias_alloc = _tensor_builder->at(bias_index).get();
  const auto frontend_layout = _current_op_seq_layout;
  const auto acl_layout = output_alloc->handle()->info()->data_layout();

  auto fn = std::make_unique<arm_compute::CLFullyConnectedReshapingLayer>(
      _tensor_builder->acl_tensor_manager()->internal_buffer_manager());

  arm_compute::CLFullyConnectedReshapingLayer::KernelType kernel_type =
      arm_compute::CLFullyConnectedReshapingLayer::KernelType::GENERAL;
  if (_ctx.at(weight_index).isConstant())
  {
    kernel_type = arm_compute::CLFullyConnectedReshapingLayer::KernelType::PREPROCESSED_WEIGHTS;
    assert(_ctx.at(weight_index).data());
  }
  fn->configure(
      input_alloc->handle(), weight_alloc->handle(), bias_alloc->handle(), output_alloc->handle(),
      needs_reshape,
      ::onert::backend::acl_common::asTensorShape(
          reshape, frontend_layout, ::onert::backend::acl_common::asRuntimeLayout(acl_layout)),
      kernel_type);

  _return_fn = std::make_unique<exec::FunctionSequence>(
      asAclClFunction(std::move(fn)),
      ActivationBuilder::generate(activation, output_alloc->handle()));
}

void KernelGenerator::visit(const ir::operation::Mul &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto lhs_index{node.getInputs().at(ir::operation::Mul::Input::LHS)};
  const auto rhs_index{node.getInputs().at(ir::operation::Mul::Input::RHS)};

  const auto activation = node.param().activation;

  auto ofm_alloc = _tensor_builder->at(ofm_index).get();
  auto lhs_alloc = _tensor_builder->at(lhs_index).get();
  auto rhs_alloc = _tensor_builder->at(rhs_index).get();

  auto fn = std::make_unique<::arm_compute::CLPixelWiseMultiplication>();

  fn->configure(lhs_alloc->handle(), rhs_alloc->handle(), ofm_alloc->handle(), 1.0, // scale
                arm_compute::ConvertPolicy::SATURATE, arm_compute::RoundingPolicy::TO_NEAREST_EVEN);

  _return_fn = std::make_unique<exec::FunctionSequence>(
      asAclClFunction(std::move(fn)), ActivationBuilder::generate(activation, ofm_alloc->handle()));
}

void KernelGenerator::visit(const ir::operation::Reduce &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Reduce::Input::INPUT)};
  const auto axes_index{node.getInputs().at(ir::operation::Reduce::Input::AXES)};
  const auto keep_dims{node.param().keep_dims};
  const auto reduce_type = node.param().reduce_type;

  auto output_alloc = _tensor_builder->at(output_index).get();
  auto input_alloc = _tensor_builder->at(input_index).get();

  // Convert to ACL axes taking into account negative values and possible duplicates.
  const auto &axes = _ctx.at(axes_index);
  const auto input_rank = _ctx.at(input_index).shape().rank();
  const auto frontend_layout = _current_op_seq_layout;
  const auto backend_layout = input_alloc->layout();

  std::unique_ptr<arm_compute::IFunction> fn;
  if (reduce_type == ir::operation::Reduce::ReduceType::MEAN)
  {
    auto l = std::make_unique<::arm_compute::CLReduceMean>();

    const auto acl_axes =
        acl_common::asCoordinates(axes, input_rank, frontend_layout, backend_layout);
    l->configure(input_alloc->handle(), acl_axes, keep_dims, output_alloc->handle());

    fn = std::move(l);
  }
  else
  {
    auto l = std::make_unique<::arm_compute::CLReduceOperation>(
        _tensor_builder->acl_tensor_manager()->internal_buffer_manager());

    const auto acl_axes = acl_common::asSet(axes, input_rank, frontend_layout, backend_layout);
    l->configure(input_alloc->handle(), output_alloc->handle(), acl_axes, keep_dims,
                 acl_common::convertReduceType(reduce_type));

    fn = std::move(l);
  }

  auto acl_fn = asAclClFunction(std::move(fn));

  _return_fn = std::move(acl_fn);
}

void KernelGenerator::visit(const ir::operation::Reshape &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Reshape::Input::INPUT)};

  auto output_alloc = _tensor_builder->at(output_index).get();
  auto input_alloc = _tensor_builder->at(input_index).get();

  // NOTE This operation must not be changed the layout from frontend to backend
  //      So, PermutationOperationPass makes layouts of frontend and backend the same.
  const auto frontend_layout = _current_op_seq_layout;
  const auto backend_layout = output_alloc->layout();
  assert((_ctx.at(input_index).shape().rank() < 4 && _ctx.at(output_index).shape().rank() < 4) ||
         frontend_layout == backend_layout);
  UNUSED_RELEASE(frontend_layout);
  UNUSED_RELEASE(backend_layout);

  auto fn = std::make_unique<::arm_compute::CLReshapeLayer>();

  fn->configure(input_alloc->handle(), output_alloc->handle());

  auto acl_fn = asAclClFunction(std::move(fn));

  _return_fn = std::move(acl_fn);
}

void KernelGenerator::visit(const ir::operation::Squeeze &node)
{
  // Squeeze is identical to reshape except that it has an optional dimensions input.
  // In addition, optional dims_index is ignored since output tensor already has squeezed shape
  // by freezer and toco
  // TODO Support multi-layout for frontend and backend
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Squeeze::Input::INPUT)};
  const auto dims{node.param().dims};
  const auto ndim{node.param().ndim};
  (void)dims;
  (void)ndim;

  auto output_alloc = _tensor_builder->at(output_index).get();
  auto input_alloc = _tensor_builder->at(input_index).get();
  auto fn = std::make_unique<arm_compute::CLReshapeLayer>();
  fn->configure(input_alloc->handle(), output_alloc->handle());
  auto acl_fn = asAclClFunction(std::move(fn));
  _return_fn = std::move(acl_fn);
}

void KernelGenerator::visit(const ir::operation::Tanh &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Tanh::Input::INPUT)};

  auto output_alloc = _tensor_builder->at(output_index).get();
  auto input_alloc = _tensor_builder->at(input_index).get();

  auto fn = std::make_unique<arm_compute::CLActivationLayer>();

  const ::arm_compute::ActivationLayerInfo act_info{
      ::arm_compute::ActivationLayerInfo::ActivationFunction::TANH, 1.0f, 1.0f};

  fn->configure(input_alloc->handle(), output_alloc->handle(), act_info);

  auto acl_fn = asAclClFunction(std::move(fn));

  _return_fn = std::move(acl_fn);
}

void KernelGenerator::visit(const ir::operation::Softmax &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Softmax::Input::INPUT)};

  const auto beta = node.param().beta;

  auto output_alloc = _tensor_builder->at(output_index).get();
  auto input_alloc = _tensor_builder->at(input_index).get();

  auto fn = std::make_unique<::arm_compute::CLSoftmaxLayer>(
      _tensor_builder->acl_tensor_manager()->internal_buffer_manager());

  fn->configure(input_alloc->handle(), output_alloc->handle(), beta);

  auto acl_fn = asAclClFunction(std::move(fn));

  _return_fn = std::move(acl_fn);
}

void KernelGenerator::visit(const ir::operation::Slice &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Slice::Input::INPUT)};
  const auto begins_index{node.getInputs().at(ir::operation::Slice::Input::BEGINS)};
  const auto sizes_index{node.getInputs().at(ir::operation::Slice::Input::SIZES)};

  auto outputData_alloc = _tensor_builder->at(output_index).get();
  auto inputData_alloc = _tensor_builder->at(input_index).get();
  const auto frontend_layout = _current_op_seq_layout;
  const auto backend_layout = inputData_alloc->layout();

  // Set initializers for indices data such as order of inputData
  int input_rank = _ctx.at(input_index).shape().rank();
  std::vector<int32_t> starts;
  std::vector<int32_t> ends;
  starts.resize(input_rank, 0);
  ends.resize(input_rank, 0);
  {
    assert(_ctx.at(begins_index).data());
    assert(_ctx.at(sizes_index).data());
    auto beginData_base = _ctx.at(begins_index).data()->base();
    auto sizeData_base = _ctx.at(sizes_index).data()->base();
    const int beginData_size = _ctx.at(begins_index).shape().num_elements();
    const int sizeData_size = _ctx.at(sizes_index).shape().num_elements();

    using ir::DataType;

    UNUSED_RELEASE(beginData_size);
    UNUSED_RELEASE(sizeData_size);

    assert(_ctx.at(begins_index).typeInfo().type() == DataType::INT32);
    assert(_ctx.at(sizes_index).typeInfo().type() == DataType::INT32);
    assert(beginData_size == input_rank);
    assert(sizeData_size == input_rank);

    assert(beginData_base != nullptr);
    for (int n = 0; n < input_rank; ++n)
    {
      auto axis = ::onert::backend::acl_common::ToARMComputeAxis(input_rank, n, frontend_layout,
                                                                 backend_layout)
                      .value();

      int32_t begin_value = *(reinterpret_cast<const int32_t *>(beginData_base) + n);
      starts[axis] = begin_value;

      int32_t size_value = *(reinterpret_cast<const int32_t *>(sizeData_base) + n);
      ends[axis] = begin_value + size_value;
    }
  }

  ::arm_compute::Coordinates starts_set;
  ::arm_compute::Coordinates ends_set;

  for (size_t i = 0; i < starts.size(); ++i)
  {
    starts_set.set(i, starts[i]);
    ends_set.set(i, ends[i]);
  }

  auto fn = std::make_unique<::arm_compute::CLSlice>();

  fn->configure(inputData_alloc->handle(), outputData_alloc->handle(), starts_set, ends_set);

  auto acl_fn = asAclClFunction(std::move(fn));

  _return_fn = std::move(acl_fn);
}

void KernelGenerator::visit(const ir::operation::StridedSlice &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::StridedSlice::Input::INPUT)};
  const auto starts_index{node.getInputs().at(ir::operation::StridedSlice::Input::STARTS)};
  const auto ends_index{node.getInputs().at(ir::operation::StridedSlice::Input::ENDS)};
  const auto strides_index{node.getInputs().at(ir::operation::StridedSlice::Input::STRIDES)};

  auto outputData_alloc = _tensor_builder->at(output_index).get();
  auto inputData_alloc = _tensor_builder->at(input_index).get();
  const auto frontend_layout = _current_op_seq_layout;
  const auto backend_layout = inputData_alloc->layout();

  // Set initializers for indices data such as order of inputData
  int input_rank = _ctx.at(input_index).shape().rank();
  std::vector<int32_t> starts;
  std::vector<int32_t> ends;
  std::vector<int32_t> strides;
  starts.resize(input_rank, 0);
  ends.resize(input_rank, 0);
  strides.resize(input_rank, 0);
  {
    assert(_ctx.at(starts_index).data());
    assert(_ctx.at(ends_index).data());
    assert(_ctx.at(strides_index).data());
    auto startData_base = _ctx.at(starts_index).data()->base();
    auto endData_base = _ctx.at(ends_index).data()->base();
    auto stridesData_base = _ctx.at(strides_index).data()->base();
    const int startData_size = _ctx.at(starts_index).shape().num_elements();
    const int endData_size = _ctx.at(ends_index).shape().num_elements();
    const int stridesData_size = _ctx.at(strides_index).shape().num_elements();

    using ir::DataType;

    UNUSED_RELEASE(startData_size);
    UNUSED_RELEASE(endData_size);
    UNUSED_RELEASE(stridesData_size);

    assert(_ctx.at(starts_index).typeInfo().type() == DataType::INT32);
    assert(_ctx.at(ends_index).typeInfo().type() == DataType::INT32);
    assert(_ctx.at(strides_index).typeInfo().type() == DataType::INT32);
    assert(startData_size == input_rank);
    assert(endData_size == input_rank);
    assert(stridesData_size == input_rank);

    assert(startData_base != nullptr);
    for (int n = 0; n < input_rank; ++n)
    {
      auto axis = ::onert::backend::acl_common::ToARMComputeAxis(input_rank, n, frontend_layout,
                                                                 backend_layout)
                      .value();

      int32_t start_value = *(reinterpret_cast<const int32_t *>(startData_base) + n);
      starts[axis] = start_value;

      int32_t end_value = *(reinterpret_cast<const int32_t *>(endData_base) + n);
      ends[axis] = end_value;

      int32_t strides_value = *(reinterpret_cast<const int32_t *>(stridesData_base) + n);
      strides[axis] = strides_value;
    }
  }

  // Set mask bits such as order of inputData
  const auto begin_mask = acl_common::ReorderBits<int32_t>(node.param().begin_mask, input_rank,
                                                           frontend_layout, backend_layout);
  const auto end_mask = acl_common::ReorderBits<int32_t>(node.param().end_mask, input_rank,
                                                         frontend_layout, backend_layout);
  const auto shrink_axis_mask = acl_common::ReorderBits<int32_t>(
      node.param().shrink_axis_mask, input_rank, frontend_layout, backend_layout);

  ::arm_compute::Coordinates starts_set;
  ::arm_compute::Coordinates ends_set;
  ::arm_compute::BiStrides strides_set;

  for (size_t i = 0; i < starts.size(); ++i)
  {
    starts_set.set(i, starts[i]);
    ends_set.set(i, ends[i]);
    strides_set.set(i, strides[i]);
  }

  auto fn = std::make_unique<::arm_compute::CLStridedSlice>();

  fn->configure(inputData_alloc->handle(), outputData_alloc->handle(), starts_set, ends_set,
                strides_set, begin_mask, end_mask, shrink_axis_mask);

  auto acl_fn = asAclClFunction(std::move(fn));

  _return_fn = std::move(acl_fn);
}

void KernelGenerator::visit(const ir::operation::Transpose &node)
{
  const auto ofm_idx{node.getOutputs().at(0)};
  const auto ifm_idx{node.getInputs().at(ir::operation::Transpose::Input::INPUT)};
  const auto &perm{node.param().perm};

  const auto rank = _ctx.at(ifm_idx).shape().rank();

  auto ofm_alloc = _tensor_builder->at(ofm_idx).get();
  auto ifm_alloc = _tensor_builder->at(ifm_idx).get();
  const auto frontend_layout = _current_op_seq_layout;
  const auto backend_layout = ifm_alloc->layout();

  std::vector<std::int32_t> pv(perm.cbegin(), perm.cend());
  // Reversed
  auto backend_pv = ::onert::backend::acl_common::getARMComputePermutationVector(
      rank, pv, frontend_layout, backend_layout);

  auto fn = std::make_unique<::arm_compute::CLPermute>();

  fn->configure(ifm_alloc->handle(), ofm_alloc->handle(), backend_pv);

  auto acl_fn = asAclClFunction(std::move(fn));

  _return_fn = std::move(acl_fn);
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

  auto fn = std::make_unique<::arm_compute::CLArithmeticAddition>();

  fn->configure(lhs_alloc->handle(), rhs_alloc->handle(), ofm_alloc->handle(),
                arm_compute::ConvertPolicy::SATURATE);

  _return_fn = std::make_unique<exec::FunctionSequence>(
      asAclClFunction(std::move(fn)), ActivationBuilder::generate(activation, ofm_alloc->handle()));
}

void KernelGenerator::visit(const ir::operation::Sub &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto lhs_index{node.getInputs().at(ir::operation::Sub::Input::LHS)};
  const auto rhs_index{node.getInputs().at(ir::operation::Sub::Input::RHS)};

  const auto activation = node.param().activation;

  auto ofm_alloc = _tensor_builder->at(ofm_index).get();
  auto lhs_alloc = _tensor_builder->at(lhs_index).get();
  auto rhs_alloc = _tensor_builder->at(rhs_index).get();

  auto fn = std::make_unique<::arm_compute::CLArithmeticSubtraction>();

  fn->configure(lhs_alloc->handle(), rhs_alloc->handle(), ofm_alloc->handle(),
                arm_compute::ConvertPolicy::SATURATE);

  _return_fn = std::make_unique<exec::FunctionSequence>(
      asAclClFunction(std::move(fn)), ActivationBuilder::generate(activation, ofm_alloc->handle()));
}

void KernelGenerator::visit(const ir::operation::Div &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto lhs_index{node.getInputs().at(ir::operation::Div::Input::LHS)};
  const auto rhs_index{node.getInputs().at(ir::operation::Div::Input::RHS)};

  const auto activation = node.param().activation;

  auto ofm_alloc = _tensor_builder->at(ofm_index).get();
  auto lhs_alloc = _tensor_builder->at(lhs_index).get();
  auto rhs_alloc = _tensor_builder->at(rhs_index).get();

  auto fn = std::make_unique<::arm_compute::CLArithmeticDivision>();

  fn->configure(lhs_alloc->handle(), rhs_alloc->handle(), ofm_alloc->handle());

  _return_fn = std::make_unique<exec::FunctionSequence>(
      asAclClFunction(std::move(fn)), ActivationBuilder::generate(activation, ofm_alloc->handle()));
}

void KernelGenerator::visit(const ir::operation::Exp &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Exp::Input::INPUT)};

  auto output_alloc = _tensor_builder->at(output_index).get();
  auto input_alloc = _tensor_builder->at(input_index).get();

  auto fn = std::make_unique<::arm_compute::CLExpLayer>();

  fn->configure(input_alloc->handle(), output_alloc->handle());

  auto acl_fn = asAclClFunction(std::move(fn));

  _return_fn = std::move(acl_fn);
}

void KernelGenerator::visit(const ir::operation::ExpandDims &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::ExpandDims::Input::INPUT)};

  auto output_alloc = _tensor_builder->at(output_index).get();
  auto input_alloc = _tensor_builder->at(input_index).get();

  auto fn = std::make_unique<::arm_compute::CLReshapeLayer>();

  fn->configure(input_alloc->handle(), output_alloc->handle());

  auto acl_fn = asAclClFunction(std::move(fn));

  _return_fn = std::move(acl_fn);
}

void KernelGenerator::visit(const ir::operation::InstanceNorm &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::InstanceNorm::Input::INPUT)};
  const auto gamma_index{node.getInputs().at(ir::operation::InstanceNorm::Input::GAMMA)};
  const auto beta_index{node.getInputs().at(ir::operation::InstanceNorm::Input::BETA)};

  auto ofm_alloc = _tensor_builder->at(ofm_index).get();
  auto ifm_alloc = _tensor_builder->at(ifm_index).get();
  auto gamma_alloc = _tensor_builder->at(gamma_index).get();
  auto beta_alloc = _tensor_builder->at(beta_index).get();
  auto epsilon = node.param().epsilon;
  auto activation = node.param().activation;

  auto fn = std::make_unique<::arm_compute::CLInstanceNormalizationLayerEx>();

  fn->configure(ifm_alloc->handle(), ofm_alloc->handle(), gamma_alloc->handle(),
                beta_alloc->handle(), epsilon);

  _return_fn = std::make_unique<exec::FunctionSequence>(
      asAclClFunction(std::move(fn)), ActivationBuilder::generate(activation, ofm_alloc->handle()));
}

void KernelGenerator::visit(const ir::operation::Logistic &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::Logistic::Input::INPUT)};

  auto ofm_alloc = _tensor_builder->at(ofm_index).get();
  auto ifm_alloc = _tensor_builder->at(ifm_index).get();

  const ::arm_compute::ActivationLayerInfo act_info{
      ::arm_compute::ActivationLayerInfo::ActivationFunction::LOGISTIC};

  auto fn = std::make_unique<::arm_compute::CLActivationLayer>();

  fn->configure(ifm_alloc->handle(), ofm_alloc->handle(), act_info);

  auto acl_fn = asAclClFunction(std::move(fn));

  _return_fn = std::move(acl_fn);
}

void KernelGenerator::visit(const ir::operation::LogicalAnd &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input0_index{node.getInputs().at(ir::operation::LogicalAnd::Input::INPUT0)};
  const auto input1_index{node.getInputs().at(ir::operation::LogicalAnd::Input::INPUT1)};

  auto output_alloc = _tensor_builder->at(output_index).get();
  auto input0_alloc = _tensor_builder->at(input0_index).get();
  auto input1_alloc = _tensor_builder->at(input1_index).get();

  auto fn = std::make_unique<::arm_compute::CLBinaryLogicalOp>();

  fn->configure(input0_alloc->handle(), input1_alloc->handle(), output_alloc->handle(),
                ::arm_compute::BinaryLogicalOperation::AND);

  auto acl_fn = asAclClFunction(std::move(fn));

  _return_fn = std::move(acl_fn);
}

void KernelGenerator::visit(const ir::operation::LSTM &node)
{
  // TODO Support dynamic rnn
  // TODO Fix subtle error in the case of non-CIFG, non-peephole and No Projection.
  const auto scratch_buffer_index{
      node.getOutputs().at(ir::operation::LSTM::Output::SCRATCH_BUFFER)};
  const auto output_state_out_index{
      node.getOutputs().at(ir::operation::LSTM::Output::OUTPUT_STATE_OUT)};
  const auto cell_state_out_index{
      node.getOutputs().at(ir::operation::LSTM::Output::CELL_STATE_OUT)};
  const auto output_index{node.getOutputs().at(ir::operation::LSTM::Output::OUTPUT)};

  const auto input_index{node.getInputs().at(ir::operation::LSTM::Input::INPUT)};
  const auto input_to_input_weights_index{
      node.getInputs().at(ir::operation::LSTM::Input::INPUT_TO_INPUT_WEIGHTS)}; // optional
  const auto input_to_forget_weights_index{
      node.getInputs().at(ir::operation::LSTM::Input::INPUT_TO_FORGET_WEIGHTS)};
  const auto input_to_cell_weights_index{
      node.getInputs().at(ir::operation::LSTM::Input::INPUT_TO_CELL_WEIGHTS)};
  const auto input_to_output_weights_index{
      node.getInputs().at(ir::operation::LSTM::Input::INPUT_TO_OUTPUT_WEIGHTS)};
  const auto recurrent_to_input_weights_index{
      node.getInputs().at(ir::operation::LSTM::Input::RECURRENT_TO_INPUT_WEIGHTS)}; // optional
  const auto recurrent_to_forget_weights_index{
      node.getInputs().at(ir::operation::LSTM::Input::RECURRENT_TO_FORGET_WEIGHTS)};
  const auto recurrent_to_cell_weights_index{
      node.getInputs().at(ir::operation::LSTM::Input::RECURRENT_TO_CELL_WEIGHTS)};
  const auto recurrent_to_output_weights_index{
      node.getInputs().at(ir::operation::LSTM::Input::RECURRENT_TO_OUTPUT_WEIGHTS)};
  const auto cell_to_input_weights_index{
      node.getInputs().at(ir::operation::LSTM::Input::CELL_TO_INPUT_WEIGHTS)}; // optional
  const auto cell_to_forget_weights_index{
      node.getInputs().at(ir::operation::LSTM::Input::CELL_TO_FORGET_WEIGHTS)}; // optional
  const auto cell_to_output_weights_index{
      node.getInputs().at(ir::operation::LSTM::Input::CELL_TO_OUTPUT_WEIGHTS)}; // optional
  const auto input_gate_bias_index{
      node.getInputs().at(ir::operation::LSTM::Input::INPUT_GATE_BIAS)};
  const auto forget_gate_bias_index{
      node.getInputs().at(ir::operation::LSTM::Input::FORGET_GATE_BIAS)};
  const auto cell_bias_index{node.getInputs().at(ir::operation::LSTM::Input::CELL_BIAS)};
  const auto output_gate_bias_index{
      node.getInputs().at(ir::operation::LSTM::Input::OUTPUT_GATE_BIAS)};
  const auto projection_weights_index{
      node.getInputs().at(ir::operation::LSTM::Input::PROJECTION_WEIGHTS)}; // optional
  const auto projection_bias_index{
      node.getInputs().at(ir::operation::LSTM::Input::PROJECTION_BIAS)}; // optional
  const auto output_state_in_index{
      node.getInputs().at(ir::operation::LSTM::Input::OUTPUT_STATE_IN)};
  const auto cell_state_in_index{node.getInputs().at(ir::operation::LSTM::Input::CELL_STATE_IN)};
  const auto cell_threshold = node.param().cell_threshold;
  const auto projection_threshold = node.param().projection_threshold;

  bool has_input_to_input_weights = _ctx.at(input_to_input_weights_index).shape().dim(0) != 0 &&
                                    _ctx.at(input_to_input_weights_index).shape().dim(1) != 0;
  bool has_recurrent_to_input_weights =
      _ctx.at(recurrent_to_input_weights_index).shape().dim(0) != 0 &&
      _ctx.at(recurrent_to_input_weights_index).shape().dim(1) != 0;
  bool has_cell_to_forget_weights = _ctx.at(cell_to_forget_weights_index).shape().dim(0) != 0;
  bool has_cell_to_output_weights = _ctx.at(cell_to_output_weights_index).shape().dim(0) != 0;
  bool has_projection_weights = _ctx.at(projection_weights_index).shape().dim(0) != 0 &&
                                _ctx.at(projection_weights_index).shape().dim(1) != 0;
  bool has_projection_bias = _ctx.at(projection_bias_index).shape().dim(0);

  // NOTE The input_to_input_weights and the recurrent_to_input_weights do not exist in CIFG.
  // true: no CIFG
  // false: CIFG
  // NOTE The cell_to_input_weights does not exist in non-peephole although regular LSTM(non-CIFG).
  bool has_cifg_param = has_input_to_input_weights && has_recurrent_to_input_weights;

  // NOTE The cell_to_forget_weights and the cell_to_output_weights exist in peephole.
  // But the cell_to_input_weights does not exist in regular CIFG although peephole.
  // true: peephole
  // false: no peephole
  bool has_peephole_param = has_cell_to_forget_weights && has_cell_to_output_weights;

  // NOTE Although the projection weights has data the projection bias may not have data.
  bool has_projection_param = has_projection_weights;

  const auto activation = node.param().activation;
  const auto cell_clip = cell_threshold;
  const auto projection_clip = projection_threshold;
  assert(cell_clip >= 0.f && projection_clip >= 0.f);

  auto scratch_buffer_alloc = _tensor_builder->at(scratch_buffer_index).get();
  auto output_state_out_alloc = _tensor_builder->at(output_state_out_index).get();
  auto cell_state_out_alloc = _tensor_builder->at(cell_state_out_index).get();
  auto output_alloc = _tensor_builder->at(output_index).get();

  auto input_alloc = _tensor_builder->at(input_index).get();

  auto input_to_forget_weights_alloc = _tensor_builder->at(input_to_forget_weights_index).get();
  auto input_to_cell_weights_alloc = _tensor_builder->at(input_to_cell_weights_index).get();
  auto input_to_output_weights_alloc = _tensor_builder->at(input_to_output_weights_index).get();
  auto recurrent_to_forget_weights_alloc =
      _tensor_builder->at(recurrent_to_forget_weights_index).get();
  auto recurrent_to_cell_weights_alloc = _tensor_builder->at(recurrent_to_cell_weights_index).get();
  auto recurrent_to_output_weights_alloc =
      _tensor_builder->at(recurrent_to_output_weights_index).get();

  auto forget_gate_bias_alloc = _tensor_builder->at(forget_gate_bias_index).get();
  auto cell_bias_alloc = _tensor_builder->at(cell_bias_index).get();
  auto output_gate_bias_alloc = _tensor_builder->at(output_gate_bias_index).get();
  auto output_state_in_alloc = _tensor_builder->at(output_state_in_index).get();
  auto cell_state_in_alloc = _tensor_builder->at(cell_state_in_index).get();

  auto act_info = ::onert::backend::acl_common::asActivationLayerInfo(activation);

  auto fn = std::make_unique<::arm_compute::CLLSTMLayer>();

  ::arm_compute::LSTMParams<::arm_compute::ICLTensor> lstm_params{};
  if (has_cifg_param)
  {
    auto input_to_input_weights_alloc =
        _tensor_builder->at(input_to_input_weights_index).get(); // optional
    auto recurrent_to_input_weights_alloc =
        _tensor_builder->at(recurrent_to_input_weights_index).get(); // optional
    auto cell_to_input_weights_handle =
        has_peephole_param ? _tensor_builder->at(cell_to_input_weights_index).get()->handle()
                           : nullptr; // optional (non-cifg && peephole)
    auto input_gate_bias_alloc = _tensor_builder->at(input_gate_bias_index).get(); // optional
    lstm_params.set_cifg_params(input_to_input_weights_alloc->handle(),
                                recurrent_to_input_weights_alloc->handle(),
                                cell_to_input_weights_handle, input_gate_bias_alloc->handle());
  }
  if (has_peephole_param)
  {
    auto cell_to_forget_weights_alloc =
        _tensor_builder->at(cell_to_forget_weights_index).get(); // optional
    auto cell_to_output_weights_alloc =
        _tensor_builder->at(cell_to_output_weights_index).get(); // optional
    lstm_params.set_peephole_params(cell_to_forget_weights_alloc->handle(),
                                    cell_to_output_weights_alloc->handle());
  }
  if (has_projection_param)
  {
    auto projection_weights_alloc = _tensor_builder->at(projection_weights_index).get(); // optional
    auto projection_bias_handle = has_projection_bias
                                      ? _tensor_builder->at(projection_bias_index).get()->handle()
                                      : nullptr; // optional
    lstm_params.set_projection_params(projection_weights_alloc->handle(), projection_bias_handle);
  }

  fn->configure(
      input_alloc->handle(), input_to_forget_weights_alloc->handle(),
      input_to_cell_weights_alloc->handle(), input_to_output_weights_alloc->handle(),
      recurrent_to_forget_weights_alloc->handle(), recurrent_to_cell_weights_alloc->handle(),
      recurrent_to_output_weights_alloc->handle(), forget_gate_bias_alloc->handle(),
      cell_bias_alloc->handle(), output_gate_bias_alloc->handle(), output_state_in_alloc->handle(),
      cell_state_in_alloc->handle(), scratch_buffer_alloc->handle(),
      output_state_out_alloc->handle(), cell_state_out_alloc->handle(), output_alloc->handle(),
      lstm_params, act_info, cell_clip, projection_clip);

  auto acl_fn = asAclClFunction(std::move(fn));

  _return_fn = std::move(acl_fn);
}

void KernelGenerator::visit(const ir::operation::Comparison &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input0_index{node.getInputs().at(ir::operation::Comparison::Input::INPUT0)};
  const auto input1_index{node.getInputs().at(ir::operation::Comparison::Input::INPUT1)};

  const auto comparison_type = node.param().comparison_type;

  auto output_alloc = _tensor_builder->at(output_index).get();
  auto input0_alloc = _tensor_builder->at(input0_index).get();
  auto input1_alloc = _tensor_builder->at(input1_index).get();

  auto fn = std::make_unique<::arm_compute::CLComparison>();

  fn->configure(input0_alloc->handle(), input1_alloc->handle(), output_alloc->handle(),
                (arm_compute::ComparisonOperation)comparison_type);

  auto acl_fn = asAclClFunction(std::move(fn));

  _return_fn = std::move(acl_fn);
}

void KernelGenerator::visit(const ir::operation::Pack &node)
{
  const auto output_index{node.getOutputs().at(0)};
  auto axis{node.param().axis};

  const auto output_rank = _ctx.at(output_index).shape().rank();

  std::vector<ir::OperandIndex> input_indexes;
  for (const auto &input_index : node.getInputs())
    input_indexes.emplace_back(input_index);

  auto output = _tensor_builder->at(output_index).get()->handle();
  std::vector<arm_compute::ICLTensor *> inputs;
  for (const auto &input_index : input_indexes)
    inputs.emplace_back(_tensor_builder->at(input_index)->handle());

  const auto frontend_layout = _current_op_seq_layout;
  const auto backend_layout = _tensor_builder->at(output_index).get()->layout();

  if (axis < 0)
    axis += output_rank;
  axis = acl_common::ToARMComputeAxis(output_rank, axis, frontend_layout, backend_layout).value();

  auto fn = std::make_unique<::arm_compute::CLStackLayer>();

  // Disable applied dim_correction
  std::vector<arm_compute::TensorShape> orig_inputs_acl_tensor_shapes;
  for (const auto &input_index : input_indexes)
  {
    size_t input_rank = _ctx.at(input_index).shape().rank();
    const auto &input_alloc = _tensor_builder->at(input_index);
    orig_inputs_acl_tensor_shapes.emplace_back(input_alloc->info()->tensor_shape());
    assert(input_rank == input_alloc->num_dimensions());
    if (input_rank != input_alloc->info()->num_dimensions())
    {
      // This means that high dimension's value is 1 and ifm tensor is applied dim_correction
      input_alloc->info()->set_tensor_shape(acl_common::asTensorShape(
          _ctx.at(input_index).shape(), _current_op_seq_layout, backend_layout, false));
    }
  }

  fn->configure(inputs, axis, output);

  // Revert disabling applied dim_correction
  assert(inputs.size() == orig_inputs_acl_tensor_shapes.size());
  for (size_t i = 0; i < inputs.size(); ++i)
  {
    inputs.at(i)->info()->set_tensor_shape(orig_inputs_acl_tensor_shapes.at(i));
  }

  _return_fn = asAclClFunction(std::move(fn));
}

void KernelGenerator::visit(const ir::operation::Permute &node)
{
  const auto ofm_idx{node.getOutputs().at(0)};
  const auto ifm_idx{node.getInputs().at(0)};
  const auto permute_type = node.getPermuteType();
  auto ofm_alloc = _tensor_builder->at(ofm_idx).get();
  auto ifm_alloc = _tensor_builder->at(ifm_idx).get();
  const auto rank = _ctx.at(ofm_idx).shape().rank();
  assert(_ctx.at(ifm_idx).shape().rank() == _ctx.at(ofm_idx).shape().rank());

  std::unique_ptr<::arm_compute::IFunction> fn;
  arm_compute::PermutationVector pv;
  if (permute_type == ir::operation::Permute::Type::NCHW_TO_NHWC && rank == 4)
  {
    // WHCN -> CWHN
    pv = arm_compute::PermutationVector{2, 0, 1};

    auto l = std::make_unique<::arm_compute::CLPermute>();

    l->configure(ifm_alloc->handle(), ofm_alloc->handle(), pv);

    fn = std::move(l);
  }
  else if (permute_type == ir::operation::Permute::Type::NHWC_TO_NCHW && rank == 4)
  {
    // CWHN -> WHCN
    pv = arm_compute::PermutationVector{1, 2, 0};

    auto l = std::make_unique<::arm_compute::CLPermute>();

    l->configure(ifm_alloc->handle(), ofm_alloc->handle(), pv);

    fn = std::move(l);
  }
  else
  {
    auto l = std::make_unique<::arm_compute::CLCopy>();

    l->configure(ifm_alloc->handle(), ofm_alloc->handle());

    fn = std::move(l);
  }

  auto acl_fn = asAclClFunction(std::move(fn));

  _return_fn = std::move(acl_fn);
}

void KernelGenerator::visit(const ir::operation::RSQRT &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::RSQRT::Input::INPUT)};

  auto ofm_alloc = _tensor_builder->at(ofm_index).get();
  auto ifm_alloc = _tensor_builder->at(ifm_index).get();

  auto fn = std::make_unique<::arm_compute::CLRsqrtLayer>();

  fn->configure(ifm_alloc->handle(), ofm_alloc->handle());

  _return_fn = asAclClFunction(std::move(fn));
}

void KernelGenerator::visit(const ir::operation::ReLU &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::ReLU::Input::INPUT)};

  auto output_alloc = _tensor_builder->at(output_index).get();
  auto input_alloc = _tensor_builder->at(input_index).get();

  auto fn = std::make_unique<arm_compute::CLActivationLayer>();

  const ::arm_compute::ActivationLayerInfo act_info{
      ::arm_compute::ActivationLayerInfo::ActivationFunction::RELU};

  fn->configure(input_alloc->handle(), output_alloc->handle(), act_info);

  auto acl_fn = asAclClFunction(std::move(fn));

  _return_fn = std::move(acl_fn);
}

void KernelGenerator::visit(const ir::operation::ResizeBilinear &node)
{
  const auto ofm_index{node.getOutputs().at(0)};

  const auto ifm_index{node.getInputs().at(ir::operation::ResizeBilinear::Input::INPUT)};

  auto ofm_alloc = _tensor_builder->at(ofm_index).get();
  auto ifm_alloc = _tensor_builder->at(ifm_index).get();

  auto fn = std::make_unique<::arm_compute::CLScale>();

  fn->configure(ifm_alloc->handle(), ofm_alloc->handle(),
                ::arm_compute::InterpolationPolicy::BILINEAR, ::arm_compute::BorderMode::REPLICATE,
                ::arm_compute::PixelValue(0.f), ::arm_compute::SamplingPolicy::TOP_LEFT);

  auto acl_fn = asAclClFunction(std::move(fn));

  _return_fn = std::move(acl_fn);
}

void KernelGenerator::visit(const ir::operation::ReLU1 &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::ReLU1::Input::INPUT)};

  auto ofm_alloc = _tensor_builder->at(ofm_index).get();
  auto ifm_alloc = _tensor_builder->at(ifm_index).get();

  const ::arm_compute::ActivationLayerInfo act_info{
      ::arm_compute::ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 1.0f, -1.0f};

  auto fn = std::make_unique<::arm_compute::CLActivationLayer>();

  fn->configure(ifm_alloc->handle(), ofm_alloc->handle(), act_info);

  auto acl_fn = asAclClFunction(std::move(fn));

  _return_fn = std::move(acl_fn);
}

void KernelGenerator::visit(const ir::operation::ReLU6 &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::ReLU6::Input::INPUT)};

  auto ofm_alloc = _tensor_builder->at(ofm_index).get();
  auto ifm_alloc = _tensor_builder->at(ifm_index).get();

  const ::arm_compute::ActivationLayerInfo act_info{
      ::arm_compute::ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.0f};

  auto fn = std::make_unique<::arm_compute::CLActivationLayer>();

  fn->configure(ifm_alloc->handle(), ofm_alloc->handle(), act_info);

  auto acl_fn = asAclClFunction(std::move(fn));

  _return_fn = std::move(acl_fn);
}

void KernelGenerator::visit(const ir::operation::RNN &node)
{
  const auto output_index{node.getOutputs().at(ir::operation::RNN::Output::OUTPUT)};
  const auto hidden_state_out_index{
      node.getOutputs().at(ir::operation::RNN::Output::HIDDEN_STATE_OUT)};

  const auto input_index{node.getInputs().at(ir::operation::RNN::Input::INPUT)};
  const auto weights_index{node.getInputs().at(ir::operation::RNN::Input::WEIGHTS)};
  const auto recurrent_weights_index{
      node.getInputs().at(ir::operation::RNN::Input::RECURRENT_WEIGHTS)};
  const auto bias_index{node.getInputs().at(ir::operation::RNN::Input::BIAS)};
  const auto hidden_state_in_index{node.getInputs().at(ir::operation::RNN::Input::HIDDEN_STATE_IN)};

  const auto activation = node.param().activation;

  auto output_alloc = _tensor_builder->at(output_index).get();
  auto hidden_state_out_alloc = _tensor_builder->at(hidden_state_out_index).get();

  auto input_alloc = _tensor_builder->at(input_index).get();
  auto weights_alloc = _tensor_builder->at(weights_index).get();
  auto recurrent_weights_alloc = _tensor_builder->at(recurrent_weights_index).get();
  auto bias_alloc = _tensor_builder->at(bias_index).get();
  auto hidden_state_in_alloc = _tensor_builder->at(hidden_state_in_index).get();
  auto act_info = ::onert::backend::acl_common::asActivationLayerInfo(activation);

  auto copy_layer = std::make_unique<::arm_compute::CLCopy>();
  copy_layer->configure(hidden_state_in_alloc->handle(), hidden_state_out_alloc->handle());
  _return_fn = asAclClFunction(std::move(copy_layer));

  auto fn = std::make_unique<::arm_compute::CLRNNLayerEx>(
      _tensor_builder->acl_tensor_manager()->internal_buffer_manager());
  fn->configure(input_alloc->handle(), weights_alloc->handle(), recurrent_weights_alloc->handle(),
                bias_alloc->handle(), hidden_state_out_alloc->handle(), output_alloc->handle(),
                act_info);
  _return_fn = asAclClFunction(std::move(fn));
}

void KernelGenerator::visit(const ir::operation::Floor &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::Floor::Input::INPUT)};

  auto ofm_alloc = _tensor_builder->at(ofm_index).get();
  auto ifm_alloc = _tensor_builder->at(ifm_index).get();

  auto fn = std::make_unique<::arm_compute::CLFloor>();

  fn->configure(ifm_alloc->handle(), ofm_alloc->handle());

  auto acl_fn = asAclClFunction(std::move(fn));

  _return_fn = std::move(acl_fn);
}

void KernelGenerator::visit(const ir::operation::SpaceToBatchND &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::SpaceToBatchND::Input::INPUT)};
  const auto block_size_index{
      node.getInputs().at(ir::operation::SpaceToBatchND::Input::BLOCK_SIZE)};
  const auto paddings_index{node.getInputs().at(ir::operation::SpaceToBatchND::Input::PADDINGS)};

  auto ofm_alloc = _tensor_builder->at(ofm_index).get();
  auto ifm_alloc = _tensor_builder->at(ifm_index).get();
  auto block_size_alloc = _tensor_builder->at(block_size_index).get();
  auto paddings_alloc = _tensor_builder->at(paddings_index).get();

  assert(_ctx.at(block_size_index).data());
  assert(_ctx.at(paddings_index).data());

  std::unique_ptr<::arm_compute::IFunction> fn;

  auto l = std::make_unique<::arm_compute::CLSpaceToBatchLayer>();
  l->configure(ifm_alloc->handle(), block_size_alloc->handle(), paddings_alloc->handle(),
               ofm_alloc->handle());
  fn = std::move(l);

  auto acl_fn = asAclClFunction(std::move(fn));

  _return_fn = std::move(acl_fn);
}

void KernelGenerator::visit(const ir::operation::SpaceToDepth &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::SpaceToDepth::Input::INPUT)};

  auto block_size = node.param().block_size;

  auto ofm_alloc = _tensor_builder->at(ofm_index).get();
  auto ifm_alloc = _tensor_builder->at(ifm_index).get();

  auto fn = std::make_unique<::arm_compute::CLSpaceToDepth>();

  fn->configure(ifm_alloc->handle(), ofm_alloc->handle(), block_size);

  auto acl_fn = asAclClFunction(std::move(fn));

  _return_fn = std::move(acl_fn);
}

void KernelGenerator::visit(const ir::operation::L2Pool2D &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::L2Pool2D::Input::INPUT)};

  const auto ifm_shape = _ctx.at(ifm_index).shape().asFeature(_current_op_seq_layout);
  const auto ofm_shape = _ctx.at(ofm_index).shape().asFeature(_current_op_seq_layout);

  uint32_t kw = node.param().kw;
  uint32_t kh = node.param().kh;
  const auto stride = node.param().stride;
  const auto padding =
      ir::calculatePadding(node.param().padding, ifm_shape, ofm_shape, stride, kw, kh);
  const auto activation = node.param().activation;

  auto ofm_alloc = _tensor_builder->at(ofm_index).get();
  auto ifm_alloc = _tensor_builder->at(ifm_index).get();

  ::arm_compute::PoolingLayerInfo info{
      ::arm_compute::PoolingType::L2, ::arm_compute::Size2D{kw, kh},
      ::onert::backend::acl_common::asPadStrideInfo(padding, stride)};

  auto fn = std::make_unique<::arm_compute::CLPoolingLayer>();

  fn->configure(ifm_alloc->handle(), ofm_alloc->handle(), info);

  _return_fn = std::make_unique<exec::FunctionSequence>(
      asAclClFunction(std::move(fn)), ActivationBuilder::generate(activation, ofm_alloc->handle()));
}

void KernelGenerator::visit(const ir::operation::EmbeddingLookup &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto lookups_index{node.getInputs().at(ir::operation::EmbeddingLookup::Input::LOOKUPS)};
  const auto values_index{node.getInputs().at(ir::operation::EmbeddingLookup::Input::VALUES)};

  auto output_alloc = _tensor_builder->at(output_index).get();
  auto lookups_alloc = _tensor_builder->at(lookups_index).get();
  auto values_alloc = _tensor_builder->at(values_index).get();

  auto fn = std::make_unique<::arm_compute::CLEmbeddingLookup>();

  fn->configure(values_alloc->handle(), output_alloc->handle(), lookups_alloc->handle());

  auto acl_fn = asAclClFunction(std::move(fn));

  _return_fn = std::move(acl_fn);
}

void KernelGenerator::visit(const ir::operation::L2Normalization &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::L2Normalization::Input::INPUT)};

  // {CL|Neon}L2Normalization performs the reduction only along dimension 0
  // L2 Normalization always performs the reduction along the depth axis
  // Thus, we repurpose {CL|Neon}NormalizationLayers to act as depthwise L2 normalizations by
  // choosing normalization parameters as below

  const auto &ifm_shape = _ctx.at(ifm_index).shape();
  // TODO Support optional constant dimension that normalization would be performed on
  const auto normalization_axis = _ctx.at(ifm_index).shape().rank() - 1;
  int32_t radius =
      2 * ifm_shape.dim(normalization_axis) + 1; // normSize = depth(last dimension) * 2 + 1
  float alpha = 1.0f;                            // In the implementation to make alpha_ become 1
  float beta = 0.5f;                             // pow(reduction, -0.5) = 1 / sqrt(reduction)
  float bias = 0.0f;                             // Don't offset the reduction.

  auto ofm_alloc = _tensor_builder->at(ofm_index).get();
  auto ifm_alloc = _tensor_builder->at(ifm_index).get();

  const auto norm_info = ::arm_compute::NormalizationLayerInfo(::arm_compute::NormType::CROSS_MAP,
                                                               radius, alpha, beta, bias, false);

  auto fn = std::make_unique<::arm_compute::CLNormalizationLayer>();

  fn->configure(ifm_alloc->handle(), ofm_alloc->handle(), norm_info);

  auto acl_fn = asAclClFunction(std::move(fn));

  _return_fn = std::move(acl_fn);
}

void KernelGenerator::visit(const ir::operation::HashtableLookup &node)
{
  const auto output_index{node.getOutputs().at(ir::operation::HashtableLookup::Output::OUTPUT)};
  const auto hits_index{node.getOutputs().at(ir::operation::HashtableLookup::Output::HITS)};

  const auto lookups_index{node.getInputs().at(ir::operation::HashtableLookup::Input::LOOKUPS)};
  const auto keys_index{node.getInputs().at(ir::operation::HashtableLookup::Input::KEYS)};
  const auto values_index{node.getInputs().at(ir::operation::HashtableLookup::Input::VALUES)};

  auto output_alloc = _tensor_builder->at(output_index).get();
  auto hits_alloc = _tensor_builder->at(hits_index).get();

  auto lookups_alloc = _tensor_builder->at(lookups_index).get();
  auto keys_alloc = _tensor_builder->at(keys_index).get();
  auto values_alloc = _tensor_builder->at(values_index).get();

  auto fn = std::make_unique<::arm_compute::CLHashtableLookup>();

  fn->configure(lookups_alloc->handle(), keys_alloc->handle(), values_alloc->handle(),
                output_alloc->handle(), hits_alloc->handle());

  auto acl_fn = asAclClFunction(std::move(fn));

  _return_fn = std::move(acl_fn);
}

void KernelGenerator::visit(const ir::operation::PReLU &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::PReLU::Input::INPUT)};
  const auto alpha_index{node.getInputs().at(ir::operation::PReLU::Input::ALPHA)};

  auto ofm_alloc = _tensor_builder->at(ofm_index).get();
  auto ifm_alloc = _tensor_builder->at(ifm_index).get();
  auto alpha_alloc = _tensor_builder->at(alpha_index).get();

  auto fn = std::make_unique<::arm_compute::CLPReLU>();

  fn->configure(ifm_alloc->handle(), alpha_alloc->handle(), ofm_alloc->handle());

  auto acl_fn = asAclClFunction(std::move(fn));

  _return_fn = std::move(acl_fn);
}

void KernelGenerator::visit(const ir::operation::TransposeConv &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ker_index{node.getInputs().at(ir::operation::TransposeConv::Input::KERNEL)};
  const auto ifm_index{node.getInputs().at(ir::operation::TransposeConv::Input::INPUT)};

  const auto ofm_shape = _ctx.at(ofm_index).shape().asFeature(_current_op_seq_layout);
  const auto ifm_shape = _ctx.at(ifm_index).shape().asFeature(_current_op_seq_layout);
  const auto ker_shape = _ctx.at(ker_index).shape().asFeature(_current_op_seq_layout);

  const auto stride = node.param().stride;

  assert((node.param().padding.type == ir::PaddingType::SAME) ||
         (node.param().padding.type == ir::PaddingType::VALID));
  auto padding = ir::calculatePadding(node.param().padding, ofm_shape, ifm_shape, stride,
                                      ker_shape.W, ker_shape.H);

  uint32_t invalid_horizontal = 0;
  uint32_t invalid_vertical = 0;
  if (node.param().padding.type == ir::PaddingType::VALID)
  {
    invalid_horizontal =
        ofm_shape.W - (1 + (ifm_shape.W - 1) * stride.horizontal) - (ker_shape.W - 1);
    invalid_vertical = ofm_shape.H - (1 + (ifm_shape.H - 1) * stride.vertical) - (ker_shape.H - 1);
  }

  auto ofm_alloc = _tensor_builder->at(ofm_index).get();
  auto ifm_alloc = _tensor_builder->at(ifm_index).get();
  auto ker_alloc = _tensor_builder->at(ker_index).get();

  const auto tconv_info = acl_common::asPadStrideInfo(padding, stride);

  auto fn = std::make_unique<::arm_compute::CLTransposeConvLayer>(
      _tensor_builder->acl_tensor_manager()->internal_buffer_manager());

  fn->configure(ifm_alloc->handle(), ker_alloc->handle(), nullptr, ofm_alloc->handle(), tconv_info,
                invalid_horizontal, invalid_vertical);

  auto acl_fn = asAclClFunction(std::move(fn));

  _return_fn = std::move(acl_fn);
}

void KernelGenerator::visit(const ir::operation::SQRT &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::SQRT::Input::INPUT)};

  auto output_alloc = _tensor_builder->at(output_index).get();
  auto input_alloc = _tensor_builder->at(input_index).get();

  const ::arm_compute::ActivationLayerInfo act_info{
      ::arm_compute::ActivationLayerInfo::ActivationFunction::SQRT};

  auto fn = std::make_unique<::arm_compute::CLActivationLayer>();

  fn->configure(input_alloc->handle(), output_alloc->handle(), act_info);

  auto acl_fn = asAclClFunction(std::move(fn));

  _return_fn = std::move(acl_fn);
}

void KernelGenerator::visit(const ir::operation::LogicalOr &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input0_index{node.getInputs().at(ir::operation::LogicalOr::Input::INPUT0)};
  const auto input1_index{node.getInputs().at(ir::operation::LogicalOr::Input::INPUT1)};

  auto output_alloc = _tensor_builder->at(output_index).get();
  auto input0_alloc = _tensor_builder->at(input0_index).get();
  auto input1_alloc = _tensor_builder->at(input1_index).get();

  auto fn = std::make_unique<::arm_compute::CLBitwiseOr>();

  fn->configure(input0_alloc->handle(), input1_alloc->handle(), output_alloc->handle());

  auto acl_fn = asAclClFunction(std::move(fn));

  _return_fn = std::move(acl_fn);
}

void KernelGenerator::visit(const ir::operation::LogicalNot &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::LogicalNot::Input::INPUT)};

  auto output_alloc = _tensor_builder->at(output_index).get();
  auto input_alloc = _tensor_builder->at(input_index).get();

  auto fn = std::make_unique<::arm_compute::CLBitwiseNot>();

  fn->configure(input_alloc->handle(), output_alloc->handle());

  auto acl_fn = asAclClFunction(std::move(fn));

  _return_fn = std::move(acl_fn);
}

void KernelGenerator::visit(const ir::operation::SquaredDifference &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto lhs_index{node.getInputs().at(ir::operation::SquaredDifference::Input::LHS)};
  const auto rhs_index{node.getInputs().at(ir::operation::SquaredDifference::Input::RHS)};

  auto ofm_alloc = _tensor_builder->at(ofm_index).get();
  auto lhs_alloc = _tensor_builder->at(lhs_index).get();
  auto rhs_alloc = _tensor_builder->at(rhs_index).get();

  auto fn = std::make_unique<::arm_compute::CLElementwiseSquaredDiff>();

  fn->configure(lhs_alloc->handle(), rhs_alloc->handle(), ofm_alloc->handle());

  auto acl_fn = asAclClFunction(std::move(fn));

  _return_fn = std::move(acl_fn);
}

void KernelGenerator::visit(const ir::operation::TopKV2 &node)
{
  const auto outputValues_index{node.getOutputs().at(ir::operation::TopKV2::Output::OUTPUT_VALUES)};
  const auto outputIndices_index{
      node.getOutputs().at(ir::operation::TopKV2::Output::OUTPUT_INDICES)};

  const auto inputData_index{node.getInputs().at(ir::operation::TopKV2::Input::INPUT)};

  // Currently, we only support the vector input.
  assert(_ctx.at(inputData_index).shape().rank() == 1 ||
         _ctx.at(inputData_index).shape().rank() == 2);

  const auto k = node.param().k;

  auto values_alloc = _tensor_builder->at(outputValues_index).get();
  auto indices_alloc = _tensor_builder->at(outputIndices_index).get();
  auto input_alloc = _tensor_builder->at(inputData_index).get();

  auto fn = std::make_unique<::arm_compute::CLTopKV2>();

  fn->configure(input_alloc->handle(), k, values_alloc->handle(), indices_alloc->handle());

  auto acl_fn = asAclClFunction(std::move(fn));

  _return_fn = std::move(acl_fn);
}

void KernelGenerator::visit(const ir::operation::Gather &node)
{
  const auto ofm_index{node.getOutputs().at(0)};

  const auto ifm_index{node.getInputs().at(ir::operation::Gather::Input::INPUT)};
  const auto indices_index{node.getInputs().at(ir::operation::Gather::Input::INDICES)};

  const auto ifm_rank = _ctx.at(ifm_index).shape().rank();
  const auto axis_raw = node.param().axis;
  const auto axis_value = (axis_raw < 0 ? (ifm_rank + axis_raw) : axis_raw);
  const int axis = ::onert::backend::acl_common::ToARMComputeAxis(ifm_rank, axis_value).value();

  auto ofm_alloc = _tensor_builder->at(ofm_index).get();
  auto ifm_alloc = _tensor_builder->at(ifm_index).get();
  auto indices_alloc = _tensor_builder->at(indices_index).get();

  // NOTE The frontend layout and backend layout must be the same for this operation.
  //      If not the same, we have to add a stage(?) to perform permutation of output tensor. It
  //      is not not efficient even if it works well. If so, it would be better to set the
  //      layout of these backend tensors to the same layout.
  //      There is also one thing we have to think about. This operation depends on the layout of
  //      a model. For example, if a model in NHWC has this operation as output rank == 4, indices
  //      rank == 2 and axis == 2, this operation should work as the axis W and C, but the axis W
  //      and C are not sequential in NCHW. So the backend in NCHW cannot handle this case.
  const auto backend_layout = ofm_alloc->layout();
  UNUSED_RELEASE(backend_layout);
  assert(backend_layout == ifm_alloc->layout());
  assert(backend_layout == indices_alloc->layout());
  assert(ifm_rank < 4 || _current_op_seq_layout == backend_layout);

  auto fn = std::make_unique<::arm_compute::CLGatherEx>();

  // input is n-D, indices k-D, output is (n + k - 1)-D
  size_t n = ifm_rank;
  assert(n == ifm_alloc->num_dimensions());
  size_t k = _ctx.at(indices_index).shape().rank();
  assert(k == indices_alloc->num_dimensions());

  // Disable applied dim_correction
  const auto orig_ifm_acl_tensor_shape = ifm_alloc->info()->tensor_shape();
  if (n != ifm_alloc->info()->num_dimensions())
  {
    // This means that high dimension's value is 1 and ifm tensor is applied dim_correction
    const auto ifm = _ctx.at(ifm_index);
    ifm_alloc->info()->set_tensor_shape(
        acl_common::asTensorShape(ifm.shape(), _current_op_seq_layout, backend_layout, false));
  }
  const auto orig_indice_acl_tensor_shape = indices_alloc->info()->tensor_shape();
  if (k != indices_alloc->info()->num_dimensions())
  {
    // This means that high dimension's value is 1 and indices tensor is applied dim_correction
    const auto indices = _ctx.at(indices_index);
    indices_alloc->info()->set_tensor_shape(
        acl_common::asTensorShape(indices.shape(), _current_op_seq_layout, backend_layout, false));
  }

  fn->configure(ifm_alloc->handle(), indices_alloc->handle(), ofm_alloc->handle(), axis);

  // Revert disabling applied dim_correction
  ifm_alloc->info()->set_tensor_shape(orig_ifm_acl_tensor_shape);
  indices_alloc->info()->set_tensor_shape(orig_indice_acl_tensor_shape);

  auto acl_fn = asAclClFunction(std::move(fn));

  _return_fn = std::move(acl_fn);
}

void KernelGenerator::visit(const ir::operation::Neg &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::Neg::Input::INPUT)};

  auto ofm_alloc = _tensor_builder->at(ofm_index).get();
  auto ifm_alloc = _tensor_builder->at(ifm_index).get();

  auto fn = std::make_unique<::arm_compute::CLNeg>();

  fn->configure(ifm_alloc->handle(), ofm_alloc->handle());

  auto acl_fn = asAclClFunction(std::move(fn));

  _return_fn = std::move(acl_fn);
}

void KernelGenerator::visit(const ir::operation::Abs &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Abs::Input::INPUT)};

  auto output_alloc = _tensor_builder->at(output_index).get();
  auto input_alloc = _tensor_builder->at(input_index).get();

  const ::arm_compute::ActivationLayerInfo act_info{
      ::arm_compute::ActivationLayerInfo::ActivationFunction::ABS};

  auto fn = std::make_unique<::arm_compute::CLActivationLayer>();

  fn->configure(input_alloc->handle(), output_alloc->handle(), act_info);

  auto acl_fn = asAclClFunction(std::move(fn));

  _return_fn = std::move(acl_fn);
}

void KernelGenerator::visit(const ir::operation::ArgMax &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::ArgMax::Input::INPUT)};

  auto ifm_shape = _ctx.at(ifm_index).shape();
  auto ofm_shape = _ctx.at(ofm_index).shape();

  assert((ifm_shape.rank() - 1) == ofm_shape.rank());

  auto ofm_alloc = _tensor_builder->at(ofm_index).get();
  auto ifm_alloc = _tensor_builder->at(ifm_index).get();
  const auto ifm_rank = _ctx.at(ifm_index).shape().rank();
  auto frontend_layout = _current_op_seq_layout;
  auto backend_layout = ifm_alloc->layout();

  int axis_value = node.param().axis;
  if (axis_value < 0)
  {
    axis_value += ifm_rank;
  }

  auto acl_axis =
      acl_common::ToARMComputeAxis(ifm_rank, axis_value, frontend_layout, backend_layout).value();

  auto fn = std::make_unique<::arm_compute::CLArgOperation>();

  fn->configure(ifm_alloc->handle(), ofm_alloc->handle(), {acl_axis},
                ::arm_compute::ArgOperation::MAX);

  auto acl_fn = asAclClFunction(std::move(fn));

  _return_fn = std::move(acl_fn);
}

void KernelGenerator::visit(const ir::operation::Dequantize &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Dequantize::Input::INPUT)};

  auto output_alloc = _tensor_builder->at(output_index).get();
  auto input_alloc = _tensor_builder->at(input_index).get();

  auto fn = std::make_unique<::arm_compute::CLCast>();

  fn->configure(input_alloc->handle(), output_alloc->handle(), arm_compute::SubDataType::NONE);

  auto acl_fn = asAclClFunction(std::move(fn));

  _return_fn = std::move(acl_fn);
}

void KernelGenerator::visit(const ir::operation::LocalResponseNormalization &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{
      node.getInputs().at(ir::operation::LocalResponseNormalization::Input::INPUT)};

  auto radius = node.param().radius;
  auto alpha = node.param().alpha;
  auto beta = node.param().beta;
  auto bias = node.param().bias;

  auto ofm_alloc = _tensor_builder->at(ofm_index).get();
  auto ifm_alloc = _tensor_builder->at(ifm_index).get();

  const auto norm_info = ::arm_compute::NormalizationLayerInfo(
      ::arm_compute::NormType::CROSS_MAP, radius * 2 + 1, alpha, beta, bias, false);

  auto fn = std::make_unique<::arm_compute::CLNormalizationLayer>();

  fn->configure(ifm_alloc->handle(), ofm_alloc->handle(), norm_info);

  auto acl_fn = asAclClFunction(std::move(fn));

  _return_fn = std::move(acl_fn);
}

void KernelGenerator::visit(const ir::operation::DepthToSpace &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::DepthToSpace::Input::INPUT)};

  auto block_size = node.param().block_size;
  assert(block_size > 0);

  auto output_alloc = _tensor_builder->at(output_index).get();
  auto input_alloc = _tensor_builder->at(input_index).get();

  auto fn = std::make_unique<::arm_compute::CLDepthToSpace>();

  fn->configure(input_alloc->handle(), output_alloc->handle(), block_size);

  auto acl_fn = asAclClFunction(std::move(fn));

  _return_fn = std::move(acl_fn);
}

void KernelGenerator::visit(const ir::operation::Split &node)
{
  const auto ifm_index{node.getInputs().at(ir::operation::Split::Input::INPUT)};

  assert(node.param().num_splits == static_cast<int>(node.getOutputs().size()));

  const auto ifm_rank = _ctx.at(ifm_index).shape().rank();
  std::vector<ir::OperandIndex> output_indexes;
  for (const auto &output : node.getOutputs())
    output_indexes.emplace_back(output);

  auto ifm_alloc = _tensor_builder->at(ifm_index).get();
  std::vector<arm_compute::ICLTensor *> output_allocs;
  for (const auto &ofm_ind : output_indexes)
    output_allocs.emplace_back(_tensor_builder->at(ofm_ind).get()->handle());

  const auto frontend_layout = _current_op_seq_layout;
  const auto backend_layout = ifm_alloc->layout();
  auto axis = node.param().axis;
  if (axis < 0)
    axis += ifm_rank;
  axis = acl_common::ToARMComputeAxis(ifm_rank, axis, frontend_layout, backend_layout).value();

  auto fn = std::make_unique<::arm_compute::CLSplit>();

  fn->configure(ifm_alloc->handle(), output_allocs, axis);

  _return_fn = asAclClFunction(std::move(fn));
}

void KernelGenerator::visit(const ir::operation::Unpack &node)
{
  const auto input_index{node.getInputs().at(ir::operation::Unpack::Input::INPUT)};
  auto axis{node.param().axis};

  const auto input_rank = _ctx.at(input_index).shape().rank();

  std::vector<ir::OperandIndex> output_indexes;
  for (const auto &output_index : node.getOutputs())
    output_indexes.emplace_back(output_index);

  auto input = _tensor_builder->at(input_index).get()->handle();
  std::vector<arm_compute::ICLTensor *> outputs;
  for (const auto &output_index : output_indexes)
    outputs.emplace_back(_tensor_builder->at(output_index)->handle());

  const auto frontend_layout = _current_op_seq_layout;
  const auto backend_layout = _tensor_builder->at(input_index).get()->layout();
  if (axis < 0)
    axis += input_rank;
  axis = acl_common::ToARMComputeAxis(input_rank, axis, frontend_layout, backend_layout).value();

  // Disable applied dim_correction
  std::vector<arm_compute::TensorShape> orig_outputs_acl_tensor_shapes;
  for (const auto &output_index : output_indexes)
  {
    size_t output_rank = _ctx.at(output_index).shape().rank();
    const auto &output_alloc = _tensor_builder->at(output_index);
    orig_outputs_acl_tensor_shapes.emplace_back(output_alloc->info()->tensor_shape());
    assert(output_rank == output_alloc->num_dimensions());
    if (output_rank != output_alloc->info()->num_dimensions())
    {
      // This means that high dimension's value is 1 and ifm tensor is applied dim_correction
      output_alloc->info()->set_tensor_shape(acl_common::asTensorShape(
          _ctx.at(output_index).shape(), _current_op_seq_layout, backend_layout, false));
    }
  }

  auto fn = std::make_unique<::arm_compute::CLUnstack>();

  fn->configure(input, outputs, axis);

  _return_fn = asAclClFunction(std::move(fn));
}

void KernelGenerator::visit(const ir::operation::Pad &node)
{
  const auto input_index{node.getInputs().at(ir::operation::Pad::Input::INPUT)};
  const auto pad_index{node.getInputs().at(ir::operation::Pad::Input::PAD)};
  const auto output_index{node.getOutputs().at(0)};
  assert(_ctx.at(pad_index).data());

  auto rank = _ctx.at(input_index).shape().rank();
  auto pad_base = _ctx.at(pad_index).data()->base();

  auto input_type = _ctx.at(input_index).typeInfo();
  auto data_type = acl_common::asDataType(input_type.type());
  auto quant_info = ::arm_compute::QuantizationInfo(input_type.scale(), input_type.offset());
  const auto pixel_value = ::arm_compute::PixelValue(0, data_type, quant_info);

  auto input = _tensor_builder->at(input_index).get()->handle();
  auto output = _tensor_builder->at(output_index).get()->handle();

  const auto frontend_layout = _current_op_seq_layout;
  const auto backend_layout = _tensor_builder->at(input_index).get()->layout();

  ::arm_compute::PaddingList padding_list;
  padding_list.resize(rank);
  for (int32_t n = 0; n < rank; ++n)
  {
    const int32_t *from = reinterpret_cast<const int32_t *>(pad_base) + (n * 2);

    const auto axis =
        acl_common::ToARMComputeAxis(rank, n, frontend_layout, backend_layout).value();
    padding_list[axis] = ::arm_compute::PaddingInfo{from[0], from[1]};
  }
  auto fn = std::make_unique<::arm_compute::CLPadLayer>();

  // Disable applied dim_correction
  size_t input_rank = _ctx.at(input_index).shape().rank();
  const auto &input_alloc = _tensor_builder->at(input_index);
  assert(input_rank == input_alloc->num_dimensions());
  if (input_rank != input_alloc->info()->num_dimensions())
  {
    // This means that high dimension's value is 1 and ifm tensor is applied dim_correction
    input_alloc->info()->set_tensor_shape(acl_common::asTensorShape(
        _ctx.at(input_index).shape(), frontend_layout, backend_layout, false));
  }

  fn->configure(input, output, padding_list, pixel_value);

  // Do not revert disabling applied dim_correction CLPadKernel has cl kernel for 4-dimension
  // It would produce a mistach of result

  _return_fn = asAclClFunction(std::move(fn));
}

void KernelGenerator::visit(const ir::operation::Min &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto lhs_index{node.getInputs().at(ir::operation::Min::Input::LHS)};
  const auto rhs_index{node.getInputs().at(ir::operation::Min::Input::RHS)};

  auto ofm_alloc = _tensor_builder->at(ofm_index).get();
  auto lhs_alloc = _tensor_builder->at(lhs_index).get();
  auto rhs_alloc = _tensor_builder->at(rhs_index).get();

  auto fn = std::make_unique<::arm_compute::CLElementwiseMin>();

  fn->configure(lhs_alloc->handle(), rhs_alloc->handle(), ofm_alloc->handle());

  auto acl_fn = asAclClFunction(std::move(fn));

  _return_fn = std::move(acl_fn);
}

void KernelGenerator::visit(const ir::operation::Max &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto lhs_index{node.getInputs().at(ir::operation::Max::Input::LHS)};
  const auto rhs_index{node.getInputs().at(ir::operation::Max::Input::RHS)};

  auto ofm_alloc = _tensor_builder->at(ofm_index).get();
  auto lhs_alloc = _tensor_builder->at(lhs_index).get();
  auto rhs_alloc = _tensor_builder->at(rhs_index).get();

  auto fn = std::make_unique<::arm_compute::CLElementwiseMax>();

  fn->configure(lhs_alloc->handle(), rhs_alloc->handle(), ofm_alloc->handle());

  auto acl_fn = asAclClFunction(std::move(fn));

  _return_fn = std::move(acl_fn);
}

void KernelGenerator::visit(const ir::operation::ConvertFp32ToFp16 &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::ConvertFp32ToFp16::Input::INPUT)};

  auto ofm_alloc = _tensor_builder->at(ofm_index).get();
  auto ifm_alloc = _tensor_builder->at(ifm_index).get();

  auto fn = std::make_unique<::arm_compute::CLDepthConvertLayer>();

  fn->configure(ifm_alloc->handle(), ofm_alloc->handle(), ::arm_compute::ConvertPolicy::SATURATE,
                0);

  auto acl_fn = asAclClFunction(std::move(fn));

  _return_fn = std::move(acl_fn);
}

void KernelGenerator::visit(const ir::operation::ConvertFp16ToFp32 &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::ConvertFp16ToFp32::Input::INPUT)};

  auto ofm_alloc = _tensor_builder->at(ofm_index).get();
  auto ifm_alloc = _tensor_builder->at(ifm_index).get();

  auto fn = std::make_unique<::arm_compute::CLDepthConvertLayer>();

  fn->configure(ifm_alloc->handle(), ofm_alloc->handle(), ::arm_compute::ConvertPolicy::SATURATE,
                0);

  auto acl_fn = asAclClFunction(std::move(fn));

  _return_fn = std::move(acl_fn);
}

} // namespace acl_cl
} // namespace backend
} // namespace onert
