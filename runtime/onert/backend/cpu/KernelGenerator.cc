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

#include "ops/ArgMinMaxLayer.h"
#include "ops/BatchToSpaceNDLayer.h"
#include "ops/BinaryArithmeticLayer.h"
#include "ops/CompareLayer.h"
#include "ops/ConcatLayer.h"
#include "ops/ConvolutionLayer.h"
#include "ops/DepthwiseConvolutionLayer.h"
#include "ops/EinsumLayer.h"
#include "ops/ElementwiseActivationLayer.h"
#include "ops/ElementwiseBinaryLayer.h"
#include "ops/ElementwiseUnaryLayer.h"
#include "ops/ExpandDimsLayer.h"
#include "ops/FillLayer.h"
#include "ops/FullyConnectedLayer.h"
#include "ops/GatherLayer.h"
#include "ops/MeanLayer.h"
#include "ops/OneHotLayer.h"
#include "ops/OperationUtils.h"
#include "ops/PackLayer.h"
#include "ops/PadLayer.h"
#include "ops/PoolLayer.h"
#include "ops/PowLayer.h"
#include "ops/RangeLayer.h"
#include "ops/RankLayer.h"
#include "ops/ReduceLayer.h"
#include "ops/ReshapeLayer.h"
#include "ops/ResizeBilinearLayer.h"
#include "ops/ReverseLayer.h"
#include "ops/SelectLayer.h"
#include "ops/ShapeLayer.h"
#include "ops/SliceLayer.h"
#include "ops/SoftMaxLayer.h"
#include "ops/StridedSliceLayer.h"
#include "ops/SpaceToBatchNDLayer.h"
#include "ops/SpaceToDepthLayer.h"
#include "ops/SplitLayer.h"
#include "ops/SplitVLayer.h"
#include "ops/TileLayer.h"
#include "ops/TransposeLayer.h"
#include "ops/UnpackLayer.h"
#include "ops/SquaredDiffLayer.h"
#include "ops/L2NormLayer.h"
#include "ops/MatrixBandPartLayer.h"
#include "ops/BatchMatMulLayer.h"
#include "ops/BroadcastToLayer.h"
#include "ops/FusedBatchNormLayer.h"
#include "ops/LogSoftMaxLayer.h"
#include "ops/StatelessRandomUniformLayer.h"

#include <backend/Backend.h>
#include <backend/IConfig.h>
#include <memory>
#include <util/Utils.h>
#include <util/logging.h>
#include <exec/DynamicShapeInference.h>

#include <stdexcept>

namespace onert
{
namespace backend
{
namespace cpu
{

namespace
{
ops::ArithmeticType
convertArithmeticType(ir::operation::BinaryArithmetic::ArithmeticType arithmetic_type_ir)
{
  switch (arithmetic_type_ir)
  {
    case ir::operation::BinaryArithmetic::ArithmeticType::ADD:
      return ops::ArithmeticType::kAdd;
    case ir::operation::BinaryArithmetic::ArithmeticType::SUB:
      return ops::ArithmeticType::kSub;
    case ir::operation::BinaryArithmetic::ArithmeticType::MUL:
      return ops::ArithmeticType::kMul;
    case ir::operation::BinaryArithmetic::ArithmeticType::DIV:
      return ops::ArithmeticType::kDiv;
    default:
      throw std::runtime_error("cpu KernelGenerator : Not supported operation yet");
  }
}

ops::ElementwiseActivationType
convertElementwiseActivationType(ir::operation::ElementwiseActivation::Type type_ir)
{
  switch (type_ir)
  {
    case ir::operation::ElementwiseActivation::Type::LOGISTIC:
      return ops::ElementwiseActivationType::kLogistic;
    case ir::operation::ElementwiseActivation::Type::RELU:
      return ops::ElementwiseActivationType::kReLU;
    case ir::operation::ElementwiseActivation::Type::TANH:
      return ops::ElementwiseActivationType::kTanh;
    default:
      throw std::runtime_error("cpu KernelGenerator : Not supported operation yet");
  }
}

ops::ElementwiseBinaryType
convertElementwiseBinaryType(ir::operation::ElementwiseBinary::ElementwiseBinaryType type_ir)
{
  switch (type_ir)
  {
    case ir::operation::ElementwiseBinary::ElementwiseBinaryType::LOGICAL_OR:
      return ops::ElementwiseBinaryType::kLogicalOr;
    case ir::operation::ElementwiseBinary::ElementwiseBinaryType::MAX:
      return ops::ElementwiseBinaryType::kMax;
    case ir::operation::ElementwiseBinary::ElementwiseBinaryType::MIN:
      return ops::ElementwiseBinaryType::kMin;
    default:
      throw std::runtime_error("cpu KernelGenerator : Not supported operation yet");
  }
}

ops::ElementwiseUnaryType convertElementwiseUnaryType(ir::operation::ElementwiseUnary::Type type_ir)
{
  switch (type_ir)
  {
    case ir::operation::ElementwiseUnary::Type::ABS:
      return ops::ElementwiseUnaryType::kAbs;
    case ir::operation::ElementwiseUnary::Type::CAST:
      return ops::ElementwiseUnaryType::kCast;
    case ir::operation::ElementwiseUnary::Type::COS:
      return ops::ElementwiseUnaryType::kCos;
    case ir::operation::ElementwiseUnary::Type::DEQUANTIZE:
      return ops::ElementwiseUnaryType::kDequantize;
    case ir::operation::ElementwiseUnary::Type::ERF:
      return ops::ElementwiseUnaryType::kErf;
    case ir::operation::ElementwiseUnary::Type::EXP:
      return ops::ElementwiseUnaryType::kExp;
    case ir::operation::ElementwiseUnary::Type::LOG:
      return ops::ElementwiseUnaryType::kLog;
    case ir::operation::ElementwiseUnary::Type::LOGICAL_NOT:
      return ops::ElementwiseUnaryType::kLogicalNot;
    case ir::operation::ElementwiseUnary::Type::NEG:
      return ops::ElementwiseUnaryType::kNeg;
    case ir::operation::ElementwiseUnary::Type::QUANTIZE:
      return ops::ElementwiseUnaryType::kQuantize;
    case ir::operation::ElementwiseUnary::Type::ROUND:
      return ops::ElementwiseUnaryType::kRound;
    case ir::operation::ElementwiseUnary::Type::RSQRT:
      return ops::ElementwiseUnaryType::kRSqrt;
    case ir::operation::ElementwiseUnary::Type::SIN:
      return ops::ElementwiseUnaryType::kSin;
    case ir::operation::ElementwiseUnary::Type::ZEROS_LIKE:
      return ops::ElementwiseUnaryType::kZerosLike;
    default:
      throw std::runtime_error("cpu KernelGenerator : Not supported operation yet");
  }
}

ops::PoolType convertPoolType(ir::operation::Pool2D::PoolType type_ir)
{
  switch (type_ir)
  {
    case ir::operation::Pool2D::PoolType::AVG:
      return ops::PoolType::kAvg;
    case ir::operation::Pool2D::PoolType::MAX:
      return ops::PoolType::kMax;
    default:
      throw std::runtime_error("cpu KernelGenerator : Not supported operation yet");
  }
}

ops::ReduceType convertReduceType(ir::operation::Reduce::ReduceType reduce_type_ir)
{
  switch (reduce_type_ir)
  {
    case ir::operation::Reduce::ReduceType::ALL:
      return ops::ReduceType::kAll;
    case ir::operation::Reduce::ReduceType::ANY:
      return ops::ReduceType::kAny;
    case ir::operation::Reduce::ReduceType::MAX:
      return ops::ReduceType::kMax;
    case ir::operation::Reduce::ReduceType::MIN:
      return ops::ReduceType::kMin;
    case ir::operation::Reduce::ReduceType::PROD:
      return ops::ReduceType::kProd;
    case ir::operation::Reduce::ReduceType::SUM:
      return ops::ReduceType::kSum;
    default:
      throw std::runtime_error("cpu KernelGenerator : Not supported operation yet");
  }
}
} // namespace

KernelGenerator::KernelGenerator(
    const ir::Operands &operands_ctx, const ir::Operations &operations_ctx,
    const std::shared_ptr<TensorBuilder> &tensor_builder,
    const std::shared_ptr<cpu_common::TensorRegistry> &tensor_reg,
    const std::shared_ptr<backend::custom::IKernelBuilder> &kernel_builder,
    const std::shared_ptr<ExternalContext> &external_context)
    : _ctx(operands_ctx), _operations_ctx{operations_ctx}, _tensor_builder(tensor_builder),
      _tensor_reg{tensor_reg}, _kernel_builder(kernel_builder),
      _current_op_seq_layout(ir::Layout::UNKNOWN), _external_context(external_context)
{
  // DO NOTHING
}

void KernelGenerator::visit(const ir::OpSequence &op_seq)
{
  assert(!_return_fn_seq);
  assert(_tensor_builder->dynamicTensorManager());
  assert(_tensor_reg);

  auto dyn_shape_inferer = std::make_shared<exec::DynamicShapeInferer>(_ctx, _tensor_reg);

  _return_fn_seq = std::make_unique<exec::FunctionSequence>();

  // Prepare to handle dynamic tensors later
  auto dyn_ctx = std::make_shared<exec::FunctionSequence::DynamicTensorCtx>();
  {
    dyn_ctx->op_seq = &op_seq;
    dyn_ctx->operations = &_operations_ctx;
    dyn_ctx->dynamic_shape_inferer = std::move(dyn_shape_inferer);
    dyn_ctx->tensor_registry = _tensor_reg;
    dyn_ctx->dynamic_tensor_manager = _tensor_builder->dynamicTensorManager();

    _return_fn_seq->dynamic_tensor_ctx(dyn_ctx);
  }
  _return_fn_seq->enableDynamicShapeInferer(true);

  _current_op_seq_layout = op_seq.getLayout();
  for (const auto &operation_idx : op_seq.operations())
  {
    const auto &node = _operations_ctx.at(operation_idx);
    node.accept(*this);
    _return_fn_seq->append(releaseFunction());

    for (const auto &ind : (node.getInputs() | ir::Remove::UNDEFINED) + node.getOutputs())
    {
      auto portable_tensor = _tensor_reg->getPortableTensor(ind);
      if (portable_tensor)
      {
        assert(portable_tensor->layout() == ir::Layout::NHWC);
      }

      auto tensor = _tensor_reg->getNativeTensor(ind);
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

  auto ofm_tensor = _tensor_reg->getPortableTensor(ofm_index);
  auto ifm_tensor = _tensor_reg->getPortableTensor(ifm_index);
  auto ker_tensor = _tensor_reg->getPortableTensor(ker_index);
  auto bias_tensor = _tensor_reg->getPortableTensor(bias_index);

  const auto stride = node.param().stride;
  const auto activation = node.param().activation;
  const auto param_padding = node.param().padding;
  const auto dilation = node.param().dilation;
  auto fn = std::make_unique<ops::ConvolutionLayer>();

  if (_ctx.at(ifm_index).info().isDynamic() || _ctx.at(ker_index).info().isDynamic())
  {
    fn->configure(ifm_tensor, ker_tensor, bias_tensor, param_padding.type, param_padding.param.left,
                  param_padding.param.right, param_padding.param.top, param_padding.param.bottom,
                  stride.horizontal, stride.vertical, dilation.width_factor, dilation.height_factor,
                  activation, ofm_tensor);

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
      ir::calculatePadding(param_padding, ifm_shape, ofm_shape, stride, ker_width, ker_height,
                           dilation.width_factor, dilation.height_factor);

  fn->configure(ifm_tensor, ker_tensor, bias_tensor, param_padding.type, padding.left,
                padding.right, padding.top, padding.bottom, stride.horizontal, stride.vertical,
                dilation.width_factor, dilation.height_factor, activation, ofm_tensor);

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

  auto ofm_tensor = _tensor_reg->getPortableTensor(ofm_index);
  auto ifm_tensor = _tensor_reg->getPortableTensor(ifm_index);
  auto ker_tensor = _tensor_reg->getPortableTensor(ker_index);
  auto bias_tensor = _tensor_reg->getPortableTensor(bias_index);

  auto fn = std::make_unique<ops::DepthwiseConvolutionLayer>();

  fn->configure(ifm_tensor, ker_tensor, bias_tensor, padding.left, padding.right, padding.top,
                padding.bottom, stride.horizontal, stride.vertical, multiplier, activation,
                ofm_tensor);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Concat &node)
{
  const auto ofm_index{node.getOutputs().at(0)};

  const auto rank = _ctx.at(ofm_index).shape().rank();
  const auto axis = ops::getAxis(rank, node.param().axis, _current_op_seq_layout);

  auto output_tensor = _tensor_reg->getPortableTensor(ofm_index);

  std::vector<const IPortableTensor *> input_tensors;
  for (auto &ifm_idx : node.getInputs())
    input_tensors.emplace_back(_tensor_reg->getPortableTensor(ifm_idx));

  auto fn = std::make_unique<ops::ConcatLayer>();

  fn->configure(input_tensors, axis, output_tensor);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::BatchToSpaceND &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::BatchToSpaceND::INPUT)};
  const auto block_size_index{node.getInputs().at(ir::operation::BatchToSpaceND::BLOCK_SIZE)};

  auto output_alloc = _tensor_reg->getPortableTensor(output_index);
  auto input_alloc = _tensor_reg->getPortableTensor(input_index);
  auto block_size_alloc = _tensor_reg->getPortableTensor(block_size_index);

  auto fn = std::make_unique<ops::BatchToSpaceNDLayer>();

  IPortableTensor *crops_alloc = nullptr;
  const auto NNApiInputs = 2;

  if (node.getInputs().size() != NNApiInputs)
  {
    const auto crops_data_index{node.getInputs().at(ir::operation::BatchToSpaceND::CROPS_DATA)};
    crops_alloc = _tensor_reg->getPortableTensor(crops_data_index);
  }

  fn->configure(input_alloc, output_alloc, block_size_alloc, crops_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Fill &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Fill::Input::INPUT)};
  const auto value_index{node.getInputs().at(ir::operation::Fill::Input::VALUE)};

  auto output_tensor = _tensor_reg->getPortableTensor(output_index);
  auto input_tensor = _tensor_reg->getPortableTensor(input_index);
  auto value_tensor = _tensor_reg->getPortableTensor(value_index);

  auto fn = std::make_unique<ops::FillLayer>();

  fn->configure(input_tensor, value_tensor, output_tensor);

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

  auto output_tensor = _tensor_reg->getPortableTensor(output_index);
  auto input_tensor = _tensor_reg->getPortableTensor(input_index);
  auto weight_tensor = _tensor_reg->getPortableTensor(weight_index);
  auto bias_tensor = bias_index.undefined() ? nullptr : _tensor_reg->getPortableTensor(bias_index);

  auto fn = std::make_unique<ops::FullyConnectedLayer>();

  fn->configure(input_tensor, weight_tensor, bias_tensor, activation, output_tensor,
                _external_context);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Reshape &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Reshape::Input::INPUT)};

  auto output_tensor = _tensor_reg->getPortableTensor(output_index);
  auto input_tensor = _tensor_reg->getPortableTensor(input_index);

  // optional 2nd input
  IPortableTensor *shape_tensor = nullptr;

  if (node.getInputs().size() == 2)
  {
    const auto shape_index{node.getInputs().at(ir::operation::Reshape::Input::SHAPE)};
    shape_tensor = _tensor_reg->getPortableTensor(shape_index);
  }

  auto fn = std::make_unique<ops::ReshapeLayer>();

  fn->configure(input_tensor, shape_tensor, output_tensor);
  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Squeeze &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Squeeze::Input::INPUT)};

  auto output_tensor = _tensor_reg->getPortableTensor(output_index);
  auto input_tensor = _tensor_reg->getPortableTensor(input_index);

  // Squeeze can share same kernel with reshape
  auto fn = std::make_unique<ops::ReshapeLayer>();

  fn->configure(input_tensor, nullptr, output_tensor);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Softmax &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Softmax::Input::INPUT)};

  const auto beta = node.param().beta;

  auto output_tensor = _tensor_reg->getPortableTensor(output_index);
  auto input_tensor = _tensor_reg->getPortableTensor(input_index);

  auto fn = std::make_unique<ops::SoftMaxLayer>();

  fn->configure(input_tensor, beta, output_tensor);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::BinaryArithmetic &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto lhs_index{node.getInputs().at(ir::operation::BinaryArithmetic::Input::LHS)};
  const auto rhs_index{node.getInputs().at(ir::operation::BinaryArithmetic::Input::RHS)};

  const auto activation = node.param().activation;

  auto ofm_tensor = _tensor_reg->getPortableTensor(ofm_index);
  auto lhs_tensor = _tensor_reg->getPortableTensor(lhs_index);
  auto rhs_tensor = _tensor_reg->getPortableTensor(rhs_index);

  auto fn = std::make_unique<ops::BinaryArithmeticLayer>();

  fn->configure(lhs_tensor, rhs_tensor, ofm_tensor, activation,
                convertArithmeticType(node.param().arithmetic_type));

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Comparison &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto lhs_index{node.getInputs().at(ir::operation::Comparison::Input::INPUT0)};
  const auto rhs_index{node.getInputs().at(ir::operation::Comparison::Input::INPUT1)};

  auto ofm_tensor = _tensor_reg->getPortableTensor(ofm_index);
  auto lhs_tensor = _tensor_reg->getPortableTensor(lhs_index);
  auto rhs_tensor = _tensor_reg->getPortableTensor(rhs_index);

  auto comparison_type = node.param().comparison_type;

  auto fn = std::make_unique<ops::CompareLayer>();

  fn->configure(lhs_tensor, rhs_tensor, comparison_type, ofm_tensor);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Gather &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Gather::Input::INPUT)};
  const auto indices_index{node.getInputs().at(ir::operation::Gather::Input::INDICES)};

  auto output_tensor = _tensor_reg->getPortableTensor(output_index);
  auto input_tensor = _tensor_reg->getPortableTensor(input_index);
  auto indices_tensor = _tensor_reg->getPortableTensor(indices_index);

  const auto backend_layout = output_tensor->layout();
  UNUSED_RELEASE(backend_layout);

  // NOTE The frontend layout and backend layout must be the same for this operation.
  //      If not the same, we have to add a stage(?) to perform permutation of output tensor. It
  //      is not not efficient even if it works well. If so, it would be better to set the
  //      layout of these backend tensors to the same layout.
  //      There is also one thing we have to think about. This operation depends on the layout of
  //      a model. For example, if a model in NHWC has this operation as output rank == 4, indices
  //      rank == 2 and axis == 2, this operation should work as the axis W and C, but the axis W
  //      and C are not sequential in NCHW. So the backend in NCHW cannot handle this case.
  assert(backend_layout == input_tensor->layout());
  assert(backend_layout == indices_tensor->layout());
  const auto &input_shape = _ctx.at(input_index).shape();
  UNUSED_RELEASE(input_shape);
  assert(input_shape.rank() < 4 || _current_op_seq_layout == backend_layout);

  const auto axis_raw = node.param().axis;
  const auto axis_value = (axis_raw < 0 ? (input_shape.rank() + axis_raw) : axis_raw);

  auto fn = std::make_unique<ops::GatherLayer>();

  fn->configure(input_tensor, indices_tensor, output_tensor, axis_value);

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

  auto output_tensor = _tensor_reg->getPortableTensor(output_index);
  auto indices_tensor = _tensor_reg->getPortableTensor(indices_index);
  auto depth_tensor = _tensor_reg->getPortableTensor(depth_index);
  auto onvalue_tensor = _tensor_reg->getPortableTensor(onvalue_index);
  auto offvalue_tensor = _tensor_reg->getPortableTensor(offvalue_index);

  assert(indices_tensor->data_type() == OperandType::INT32);
  assert(axis <= static_cast<int>(indices_tensor->num_dimensions()));

  auto fn = std::make_unique<ops::OneHotLayer>();

  fn->configure(indices_tensor, depth_tensor, onvalue_tensor, offvalue_tensor, output_tensor, axis);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Einsum &node)
{
  const auto ofm_index{node.getOutputs().at(0)};

  auto output_tensor = _tensor_reg->getPortableTensor(ofm_index);
  std::vector<const IPortableTensor *> input_tensors;
  for (auto &ifm_idx : node.getInputs())
    input_tensors.emplace_back(_tensor_reg->getPortableTensor(ifm_idx));

  const auto equation = node.param().equation;

  auto fn = std::make_unique<ops::EinsumLayer>();

  fn->configure(input_tensors, equation, output_tensor);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Custom &node)
{
  auto fill_op_info = [&](const ir::OperandIndexSequence &opSeq,
                          std::vector<custom::TypeInfo> &types,
                          std::vector<IPortableTensor *> &tensors) {
    for (auto &idx : opSeq)
    {
      const auto &operand = _ctx.at(idx);
      // TODO make sure using `_current_op_seq_layout` is correct for custom operations
      types.emplace_back(custom::TypeInfo{operand.shape(), operand.typeInfo().type()});
      auto in_tensor = _tensor_reg->getPortableTensor(idx);
      tensors.emplace_back(in_tensor);
    }
  };

  backend::custom::CustomKernelConfigParams params{};

  fill_op_info(node.getInputs(), params.input_types, params.input_tensors);
  fill_op_info(node.getOutputs(), params.output_types, params.output_tensors);

  params.userdata = node.userdata().data;
  params.userdata_size = node.userdata().size;

  auto fn = _kernel_builder->buildKernel(node.id(), std::move(params));

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::ElementwiseActivation &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::ElementwiseActivation::Input::INPUT)};

  auto output_tensor = _tensor_reg->getPortableTensor(output_index);
  auto input_tensor = _tensor_reg->getPortableTensor(input_index);

  auto fn = std::make_unique<ops::ElementwiseActivationLayer>();

  fn->configure(input_tensor, output_tensor, node.param().alpha, node.param().beta,
                convertElementwiseActivationType(node.param().op_type));

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::ElementwiseBinary &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto lhs_index{node.getInputs().at(ir::operation::ElementwiseBinary::Input::LHS)};
  const auto rhs_index{node.getInputs().at(ir::operation::ElementwiseBinary::Input::RHS)};

  auto output_tensor = _tensor_reg->getPortableTensor(output_index);
  auto lhs_tensor = _tensor_reg->getPortableTensor(lhs_index);
  auto rhs_tensor = _tensor_reg->getPortableTensor(rhs_index);

  auto fn = std::make_unique<ops::ElementwiseBinaryLayer>();

  fn->configure(lhs_tensor, rhs_tensor, output_tensor,
                convertElementwiseBinaryType(node.param().op_type));

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::ElementwiseUnary &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::ElementwiseUnary::Input::INPUT)};

  auto output_tensor = _tensor_reg->getPortableTensor(output_index);
  auto input_tensor = _tensor_reg->getPortableTensor(input_index);

  auto fn = std::make_unique<ops::ElementwiseUnaryLayer>();

  fn->configure(input_tensor, output_tensor, convertElementwiseUnaryType(node.param().op_type));

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::ExpandDims &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::ExpandDims::Input::INPUT)};
  const auto axis_index{node.getInputs().at(ir::operation::ExpandDims::Input::AXIS)};

  auto output_tensor = _tensor_reg->getPortableTensor(output_index);
  auto input_tensor = _tensor_reg->getPortableTensor(input_index);
  auto axis_tensor = _tensor_reg->getPortableTensor(axis_index);

  auto fn = std::make_unique<ops::ExpandDimsLayer>();

  fn->configure(input_tensor, axis_tensor, output_tensor);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Pack &node)
{
  const auto ofm_index{node.getOutputs().at(0)};

  const auto rank = _ctx.at(ofm_index).shape().rank();
  const auto axis = ops::getAxis(rank, node.param().axis, _current_op_seq_layout);

  assert(-rank <= axis && axis < rank);

  auto output_tensor = _tensor_reg->getPortableTensor(ofm_index);

  std::vector<const IPortableTensor *> input_tensors;
  for (auto &ifm_idx : node.getInputs())
    input_tensors.emplace_back(_tensor_reg->getPortableTensor(ifm_idx));

  auto fn = std::make_unique<ops::PackLayer>();

  fn->configure(input_tensors, axis, output_tensor);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Unpack &node)
{
  const auto input_index{node.getInputs().at(0)};

  const auto rank = _ctx.at(input_index).shape().rank();
  const auto axis = ops::getAxis(rank, node.param().axis, _current_op_seq_layout);

  assert(rank == 0 || (-rank <= axis && axis < rank));

  auto input_tensor = _tensor_reg->getPortableTensor(input_index);

  std::vector<IPortableTensor *> output_tensors;
  for (auto &output_idx : node.getOutputs())
    output_tensors.emplace_back(_tensor_reg->getPortableTensor(output_idx));

  auto fn = std::make_unique<ops::UnpackLayer>();

  uint32_t axis_resolved = (axis < 0 ? axis + rank : axis);

  fn->configure(input_tensor, axis_resolved, node.param().num, output_tensors);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Pad &node)
{
  const auto input_index{node.getInputs().at(ir::operation::Pad::Input::INPUT)};
  const auto pad_index{node.getInputs().at(ir::operation::Pad::Input::PAD)};
  const auto output_index{node.getOutputs().at(0)};
  assert(_ctx.at(pad_index).data());

  auto input = _tensor_reg->getPortableTensor(input_index);
  auto output = _tensor_reg->getPortableTensor(output_index);
  auto pad_rank = _ctx.at(pad_index).shape().dim(0);
  auto pad_base = reinterpret_cast<const int32_t *>(_ctx.at(pad_index).data()->base());

  auto fn = std::make_unique<ops::PadLayer>();

  bool isPadV2 = node.getInputs().size() == 3 ? true : false;
  const void *value = nullptr;

  if (isPadV2)
  {
    const auto value_index{node.getInputs().at(ir::operation::Pad::Input::VALUE)};
    value = reinterpret_cast<const void *>(_ctx.at(value_index).data()->base());
  }

  fn->configure(input, output, pad_base, pad_rank, value);
  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Transpose &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Transpose::Input::INPUT)};
  const auto perm_index{node.getInputs().at(ir::operation::Transpose::Input::PERMUTATION)};

  auto output_tensor = _tensor_reg->getPortableTensor(output_index);
  auto input_tensor = _tensor_reg->getPortableTensor(input_index);
  auto perm_tensor = _tensor_reg->getPortableTensor(perm_index);

  auto fn = std::make_unique<ops::TransposeLayer>();

  fn->configure(input_tensor, perm_tensor, output_tensor);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Reduce &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Reduce::Input::INPUT)};
  const auto axes_index{node.getInputs().at(ir::operation::Reduce::Input::AXES)};

  const auto keep_dims = node.param().keep_dims;
  auto output_tensor = _tensor_reg->getPortableTensor(output_index);
  auto input_tensor = _tensor_reg->getPortableTensor(input_index);
  auto axes_tensor = _tensor_reg->getPortableTensor(axes_index);

  if (node.param().reduce_type == ir::operation::Reduce::ReduceType::MEAN)
  {
    auto fn = std::make_unique<ops::MeanLayer>();

    fn->configure(input_tensor, axes_tensor, output_tensor, keep_dims);

    _return_fn = std::move(fn);
  }
  else
  {
    auto fn = std::make_unique<ops::ReduceLayer>();

    const auto reduce_type = convertReduceType(node.param().reduce_type);
    fn->configure(input_tensor, axes_tensor, output_tensor, reduce_type, keep_dims);

    _return_fn = std::move(fn);
  }
}

void KernelGenerator::visit(const ir::operation::Select &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto condition_index{node.getInputs().at(ir::operation::Select::Input::CONDITION)};
  const auto true_index{node.getInputs().at(ir::operation::Select::Input::INPUT_TRUE)};
  const auto false_index{node.getInputs().at(ir::operation::Select::Input::INPUT_FALSE)};

  auto output_tensor = _tensor_reg->getPortableTensor(output_index);
  auto condition_tensor = _tensor_reg->getPortableTensor(condition_index);
  auto true_tensor = _tensor_reg->getPortableTensor(true_index);
  auto false_tensor = _tensor_reg->getPortableTensor(false_index);

  auto fn = std::make_unique<ops::SelectLayer>();

  fn->configure(condition_tensor, true_tensor, false_tensor, output_tensor);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Slice &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Slice::Input::INPUT)};
  const auto begins_index{node.getInputs().at(ir::operation::Slice::Input::BEGINS)};
  const auto sizes_index{node.getInputs().at(ir::operation::Slice::Input::SIZES)};

  auto output_tensor = _tensor_reg->getPortableTensor(output_index);
  auto input_tensor = _tensor_reg->getPortableTensor(input_index);
  auto begins_tensor = _tensor_reg->getPortableTensor(begins_index);
  auto sizes_tensor = _tensor_reg->getPortableTensor(sizes_index);

  auto fn = std::make_unique<ops::SliceLayer>();

  fn->configure(input_tensor, begins_tensor, sizes_tensor, output_tensor);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::StridedSlice &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::StridedSlice::Input::INPUT)};
  const auto starts_index{node.getInputs().at(ir::operation::StridedSlice::Input::STARTS)};
  const auto ends_index{node.getInputs().at(ir::operation::StridedSlice::Input::ENDS)};
  const auto strides_index{node.getInputs().at(ir::operation::StridedSlice::Input::STRIDES)};

  auto output_tensor = _tensor_reg->getPortableTensor(output_index);
  auto input_tensor = _tensor_reg->getPortableTensor(input_index);
  auto starts_tensor = _tensor_reg->getPortableTensor(starts_index);
  auto ends_tensor = _tensor_reg->getPortableTensor(ends_index);
  auto strides_tensor = _tensor_reg->getPortableTensor(strides_index);

  auto begin_mask = node.param().begin_mask;
  auto end_mask = node.param().end_mask;
  auto shrink_axis_mask = node.param().shrink_axis_mask;

  auto fn = std::make_unique<ops::StridedSliceLayer>();

  fn->configure(input_tensor, starts_tensor, ends_tensor, strides_tensor, output_tensor, begin_mask,
                end_mask, shrink_axis_mask);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Split &node)
{
  const auto num_splits = node.param().num_splits;
  assert(num_splits == static_cast<int>(node.getOutputs().size()));

  const auto input_idx{node.getInputs().at(ir::operation::Split::Input::INPUT)};
  const auto axis_idx{node.getInputs().at(ir::operation::Split::Input::AXIS)};

  auto in_tensor = _tensor_reg->getPortableTensor(input_idx);
  auto axis_tensor = _tensor_reg->getPortableTensor(axis_idx);

  std::vector<IPortableTensor *> out_tensors;
  for (auto &output_idx : node.getOutputs())
    out_tensors.emplace_back(_tensor_reg->getPortableTensor(output_idx));

  auto fn = std::make_unique<ops::SplitLayer>();

  fn->configure(in_tensor, axis_tensor, num_splits, out_tensors);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Shape &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::Shape::Input::INPUT)};

  auto ofm_tensor = _tensor_reg->getPortableTensor(ofm_index);
  auto ifm_tensor = _tensor_reg->getPortableTensor(ifm_index);

  auto fn = std::make_unique<ops::ShapeLayer>();

  fn->configure(ifm_tensor, ofm_tensor);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::ResizeBilinear &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::ResizeBilinear::INPUT)};

  auto align_corners = node.param().align_corners;
  auto half_pixel_centers = node.param().half_pixel_centers;

  auto output_tensor = _tensor_reg->getPortableTensor(output_index);
  auto input_tensor = _tensor_reg->getPortableTensor(input_index);

  auto fn = std::make_unique<ops::ResizeBilinearLayer>();

  if (node.getInputs().size() == 1)
  {
    fn->configure(input_tensor, output_tensor, node.param().height_out, node.param().width_out,
                  align_corners, half_pixel_centers);
  }
  else
  {
    assert(node.getInputs().size() == 2);
    const auto size_index{node.getInputs().at(ir::operation::ResizeBilinear::SIZE)};
    auto size_tensor = _tensor_reg->getPortableTensor(size_index);
    if (size_tensor->is_constant())
    {
      auto size_vec = _ctx.at(size_index).asVector<int32_t>();
      const auto height_out = size_vec[0];
      const auto width_out = size_vec[1];
      fn->configure(input_tensor, output_tensor, height_out, width_out, align_corners,
                    half_pixel_centers);
    }
    else
    {
      fn->configure(input_tensor, output_tensor, size_tensor, align_corners, half_pixel_centers);
    }
  }

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Reverse &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Reverse::INPUT)};
  const auto axis_index{node.getInputs().at(ir::operation::Reverse::AXIS)};

  auto output_tensor = _tensor_reg->getPortableTensor(output_index);
  auto input_tensor = _tensor_reg->getPortableTensor(input_index);
  auto axis_tensor = _tensor_reg->getPortableTensor(axis_index);

  auto fn = std::make_unique<ops::ReverseLayer>();

  fn->configure(input_tensor, axis_tensor, output_tensor);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::ArgMax &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::ArgMax::INPUT)};
  const auto axis_index{node.getInputs().at(ir::operation::ArgMax::AXIS)};

  auto output_tensor = _tensor_reg->getPortableTensor(output_index);
  auto input_tensor = _tensor_reg->getPortableTensor(input_index);
  auto axis_tensor = _tensor_reg->getPortableTensor(axis_index);

  auto fn = std::make_unique<ops::ArgMinMaxLayer>();

  fn->configure(input_tensor, output_tensor, axis_tensor, /* is_arg_max */ true);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Pool2D &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::Pool2D::Input::INPUT)};

  const auto kh = node.param().kh;
  const auto kw = node.param().kw;
  const auto stride = node.param().stride;
  const auto ifm_shape = _ctx.at(ifm_index).shape().asFeature(_current_op_seq_layout);
  const auto ofm_shape = _ctx.at(ofm_index).shape().asFeature(_current_op_seq_layout);
  const auto padding =
      ir::calculatePadding(node.param().padding, ifm_shape, ofm_shape, stride, kw, kh);
  const auto activation = node.param().activation;

  auto ofm_tensor = _tensor_reg->getPortableTensor(ofm_index);
  auto ifm_tensor = _tensor_reg->getPortableTensor(ifm_index);

  auto fn = std::make_unique<ops::PoolLayer>();

  fn->configure(ifm_tensor, padding.left, padding.right, padding.top, padding.bottom,
                stride.horizontal, stride.vertical, kw, kh, activation, ofm_tensor,
                convertPoolType(node.param().op_type));

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Pow &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto lhs_index{node.getInputs().at(ir::operation::Pow::LHS)};
  const auto rhs_index{node.getInputs().at(ir::operation::Pow::RHS)};

  auto output_tensor = _tensor_reg->getPortableTensor(output_index);
  auto lhs_tensor = _tensor_reg->getPortableTensor(lhs_index);
  auto rhs_tensor = _tensor_reg->getPortableTensor(rhs_index);

  auto fn = std::make_unique<ops::PowLayer>();

  fn->configure(lhs_tensor, rhs_tensor, ir::Activation::NONE, output_tensor);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::L2Normalization &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(0)};

  auto output_alloc = _tensor_reg->getPortableTensor(output_index);
  auto input_alloc = _tensor_reg->getPortableTensor(input_index);

  auto fn = std::make_unique<ops::L2NormLayer>();

  fn->configure(input_alloc, output_alloc);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Range &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto start_index{node.getInputs().at(ir::operation::Range::START)};
  const auto limit_index{node.getInputs().at(ir::operation::Range::LIMIT)};
  const auto delta_index{node.getInputs().at(ir::operation::Range::DELTA)};

  auto output_tensor = _tensor_reg->getPortableTensor(output_index);
  auto start_tensor = _tensor_reg->getPortableTensor(start_index);
  auto limit_tensor = _tensor_reg->getPortableTensor(limit_index);
  auto delta_tensor = _tensor_reg->getPortableTensor(delta_index);

  auto fn = std::make_unique<ops::RangeLayer>();

  fn->configure(start_tensor, limit_tensor, delta_tensor, output_tensor);
  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Rank &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::Shape::Input::INPUT)};

  auto ofm_tensor = _tensor_reg->getPortableTensor(ofm_index);
  auto ifm_tensor = _tensor_reg->getPortableTensor(ifm_index);

  auto fn = std::make_unique<ops::RankLayer>();

  fn->configure(ifm_tensor, ofm_tensor);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::SquaredDifference &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto lhs_index{node.getInputs().at(ir::operation::SquaredDifference::Input::LHS)};
  const auto rhs_index{node.getInputs().at(ir::operation::SquaredDifference::Input::RHS)};

  auto ofm_tensor = _tensor_reg->getPortableTensor(ofm_index);
  auto lhs_tensor = _tensor_reg->getPortableTensor(lhs_index);
  auto rhs_tensor = _tensor_reg->getPortableTensor(rhs_index);

  auto fn = std::make_unique<ops::SqDiffLayer>();

  fn->configure(lhs_tensor, rhs_tensor, ofm_tensor);
  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Tile &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Tile::INPUT)};
  const auto multiples_index{node.getInputs().at(ir::operation::Tile::MULTIPLES)};

  auto output_tensor = _tensor_reg->getPortableTensor(output_index);
  auto input_tensor = _tensor_reg->getPortableTensor(input_index);
  auto multiples_tensor = _tensor_reg->getPortableTensor(multiples_index);

  auto fn = std::make_unique<ops::TileLayer>();

  fn->configure(input_tensor, multiples_tensor, output_tensor);
  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::MatrixBandPart &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::MatrixBandPart::INPUT)};
  const auto num_lower_index{node.getInputs().at(ir::operation::MatrixBandPart::NUM_LOWER_DIAG)};
  const auto num_upper_index{node.getInputs().at(ir::operation::MatrixBandPart::NUM_UPPER_DIAG)};

  auto output_tensor = _tensor_reg->getPortableTensor(output_index);
  auto input_tensor = _tensor_reg->getPortableTensor(input_index);
  auto num_lower_tensor = _tensor_reg->getPortableTensor(num_lower_index);
  auto num_upper_tensor = _tensor_reg->getPortableTensor(num_upper_index);

  auto fn = std::make_unique<ops::MatrixBandPartLayer>();

  fn->configure(input_tensor, num_lower_tensor, num_upper_tensor, output_tensor);
  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::BatchMatMul &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto lhs_index{node.getInputs().at(ir::operation::BatchMatMul::LHS)};
  const auto rhs_index{node.getInputs().at(ir::operation::BatchMatMul::RHS)};

  auto output_tensor = _tensor_reg->getPortableTensor(output_index);
  auto lhs_tensor = _tensor_reg->getPortableTensor(lhs_index);
  auto rhs_tensor = _tensor_reg->getPortableTensor(rhs_index);

  const auto adj_x = node.param().adj_x;
  const auto adj_y = node.param().adj_y;

  auto fn = std::make_unique<ops::BatchMatMulLayer>();

  fn->configure(lhs_tensor, rhs_tensor, adj_x, adj_y, output_tensor);
  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::BroadcastTo &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::BroadcastTo::INPUT)};
  const auto shape_index{node.getInputs().at(ir::operation::BroadcastTo::SHAPE)};

  auto output_tensor = _tensor_reg->getPortableTensor(output_index);
  auto input_tensor = _tensor_reg->getPortableTensor(input_index);
  auto shape_tensor = _tensor_reg->getPortableTensor(shape_index);

  auto fn = std::make_unique<ops::BroadcastToLayer>();

  fn->configure(input_tensor, shape_tensor, output_tensor);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::FusedBatchNorm &node)
{
  const auto ofm_index{node.getOutputs().at(0)};

  auto output_tensor = _tensor_reg->getPortableTensor(ofm_index);
  std::vector<const IPortableTensor *> input_tensors;
  for (auto &ifm_idx : node.getInputs())
    input_tensors.emplace_back(_tensor_reg->getPortableTensor(ifm_idx));

  const auto epsilon = node.param().epsilon;
  const auto is_training = node.param().is_training;
  const auto data_format = node.param().data_format;

  auto fn = std::make_unique<ops::FusedBatchNormLayer>();

  fn->configure(input_tensors, epsilon, is_training, data_format, output_tensor);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::LogSoftmax &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::LogSoftmax::Input::INPUT)};

  const auto beta = node.param().beta;
  const auto axis = node.param().axis;

  auto output_tensor = _tensor_reg->getPortableTensor(output_index);
  auto input_tensor = _tensor_reg->getPortableTensor(input_index);

  auto fn = std::make_unique<ops::LogSoftMaxLayer>();

  fn->configure(input_tensor, beta, axis, output_tensor);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::SpaceToBatchND &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::SpaceToBatchND::INPUT)};
  const auto block_shape_index{node.getInputs().at(ir::operation::SpaceToBatchND::BLOCK_SIZE)};
  const auto padding_index{node.getInputs().at(ir::operation::SpaceToBatchND::PADDINGS)};

  auto output_tensor = _tensor_reg->getPortableTensor(output_index);
  auto input_tensor = _tensor_reg->getPortableTensor(input_index);
  auto block_shape_tensor = _tensor_reg->getPortableTensor(block_shape_index);
  auto padding_tensor = _tensor_reg->getPortableTensor(padding_index);

  auto fn = std::make_unique<ops::SpaceToBatchNDLayer>();

  fn->configure(input_tensor, block_shape_tensor, padding_tensor, output_tensor);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::SpaceToDepth &node)
{
  const auto input_index{node.getInputs().at(ir::operation::SpaceToDepth::Input::INPUT)};
  const auto output_index{node.getOutputs().at(0)};
  auto block_size = node.param().block_size;

  auto input_tensor = _tensor_reg->getPortableTensor(input_index);
  auto output_tensor = _tensor_reg->getPortableTensor(output_index);

  auto fn = std::make_unique<ops::SpaceToDepthLayer>();

  fn->configure(input_tensor, block_size, output_tensor);
  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::StatelessRandomUniform &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto shape_index{node.getInputs().at(ir::operation::StatelessRandomUniform::SHAPE)};
  const auto seed_index{node.getInputs().at(ir::operation::StatelessRandomUniform::SEED)};

  auto output_alloc = _tensor_reg->getPortableTensor(output_index);
  auto shape_alloc = _tensor_reg->getPortableTensor(shape_index);
  auto seed_alloc = _tensor_reg->getPortableTensor(seed_index);

  auto fn = std::make_unique<ops::StatelessRandomUniformLayer>();

  fn->configure(shape_alloc, seed_alloc, output_alloc);
  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::SplitV &node)
{
  const auto num_splits = node.param().num_splits;
  assert(num_splits == static_cast<int>(node.getOutputs().size()));

  const auto input_idx{node.getInputs().at(ir::operation::SplitV::Input::INPUT)};
  const auto size_splits{node.getInputs().at(ir::operation::SplitV::Input::SIZE_SPLITS)};
  const auto split_dim{node.getInputs().at(ir::operation::SplitV::Input::SPLIT_DIM)};

  auto in_tensor = _tensor_reg->getPortableTensor(input_idx);
  auto in_size_splits = _tensor_reg->getPortableTensor(size_splits);
  auto in_split_dim = _tensor_reg->getPortableTensor(split_dim);

  std::vector<IPortableTensor *> out_tensors;
  for (auto &output_idx : node.getOutputs())
    out_tensors.emplace_back(_tensor_reg->getPortableTensor(output_idx));

  auto fn = std::make_unique<ops::SplitVLayer>();

  fn->configure(in_tensor, in_size_splits, in_split_dim, num_splits, out_tensors);

  _return_fn = std::move(fn);
}

} // namespace cpu
} // namespace backend
} // namespace onert
