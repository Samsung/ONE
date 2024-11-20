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

#include "ops/AddNLayer.h"
#include "ops/ArgMinMaxLayer.h"
#include "ops/BatchToSpaceNDLayer.h"
#include "ops/BinaryArithmeticLayer.h"
#include "ops/CompareLayer.h"
#include "ops/ConcatLayer.h"
#include "ops/ConvolutionLayer.h"
#include "ops/DepthToSpaceLayer.h"
#include "ops/DepthwiseConvolutionLayer.h"
#include "ops/EinsumLayer.h"
#include "ops/ElementwiseActivationLayer.h"
#include "ops/ElementwiseBinaryLayer.h"
#include "ops/ElementwiseUnaryLayer.h"
#include "ops/ExpandDimsLayer.h"
#include "ops/FillLayer.h"
#include "ops/FullyConnectedLayer.h"
#include "ops/GatherLayer.h"
#include "ops/LSTMLayer.h"
#include "ops/MeanLayer.h"
#include "ops/DetectionPostProcessLayer.h"
#include "ops/OneHotLayer.h"
#include "ops/OperationUtils.h"
#include "ops/PackLayer.h"
#include "ops/PadLayer.h"
#include "ops/PoolLayer.h"
#include "ops/PowLayer.h"
#include "ops/QuantizeLayer.h"
#include "ops/RangeLayer.h"
#include "ops/RankLayer.h"
#include "ops/ReduceLayer.h"
#include "ops/ReshapeLayer.h"
#include "ops/ResizeBilinearLayer.h"
#include "ops/ReverseLayer.h"
#include "ops/RoPELayer.h"
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
#include "ops/RmsNormLayer.h"

#include <backend/Backend.h>
#include <backend/IConfig.h>
#include <memory>
#include <util/Utils.h>
#include <util/logging.h>
#include <exec/DynamicShapeInferer.h>

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
    case ir::operation::ElementwiseActivation::Type::ELU:
      return ops::ElementwiseActivationType::kElu;
    case ir::operation::ElementwiseActivation::Type::LOGISTIC:
      return ops::ElementwiseActivationType::kLogistic;
    case ir::operation::ElementwiseActivation::Type::RELU:
      return ops::ElementwiseActivationType::kReLU;
    case ir::operation::ElementwiseActivation::Type::TANH:
      return ops::ElementwiseActivationType::kTanh;
    case ir::operation::ElementwiseActivation::Type::LEAKY_RELU:
      return ops::ElementwiseActivationType::kLeakyReLU;
    default:
      throw std::runtime_error("cpu KernelGenerator : Not supported operation yet");
  }
}

ops::ElementwiseBinaryType
convertElementwiseBinaryType(ir::operation::ElementwiseBinary::ElementwiseBinaryType type_ir)
{
  switch (type_ir)
  {
    case ir::operation::ElementwiseBinary::ElementwiseBinaryType::FLOOR_DIV:
      return ops::ElementwiseBinaryType::kFloorDiv;
    case ir::operation::ElementwiseBinary::ElementwiseBinaryType::FLOOR_MOD:
      return ops::ElementwiseBinaryType::kFloorMod;
    case ir::operation::ElementwiseBinary::ElementwiseBinaryType::LOGICAL_AND:
      return ops::ElementwiseBinaryType::kLogicalAnd;
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
    case ir::operation::ElementwiseUnary::Type::FLOOR:
      return ops::ElementwiseUnaryType::kFloor;
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
    case ir::operation::ElementwiseUnary::Type::SQRT:
      return ops::ElementwiseUnaryType::kSqrt;
    case ir::operation::ElementwiseUnary::Type::SQUARE:
      return ops::ElementwiseUnaryType::kSquare;
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
  const ir::Graph &graph, const std::shared_ptr<TensorBuilder> &tensor_builder,
  const std::shared_ptr<basic::TensorRegistry> &tensor_reg,
  const std::shared_ptr<backend::custom::IKernelBuilder> &kernel_builder,
  const std::shared_ptr<ExternalContext> &external_context)
  : basic::KernelGeneratorBase{graph}, _ctx(graph.operands()), _operations_ctx{graph.operations()},
    _tensor_builder(tensor_builder), _tensor_reg{tensor_reg}, _kernel_builder(kernel_builder),
    _external_context(external_context)
{
  // DO NOTHING
}

std::unique_ptr<exec::FunctionSequence> KernelGenerator::generate(ir::OperationIndex ind)
{
  auto ret = std::make_unique<exec::FunctionSequence>();

  assert(_tensor_builder->dynamicTensorManager());
  assert(_tensor_reg);

  // Prepare to handle dynamic tensors later
  auto dyn_ctx = std::make_shared<exec::FunctionSequence::DynamicTensorCtx>();
  {
    dyn_ctx->op = &_operations_ctx.at(ind);
    dyn_ctx->dynamic_shape_inferer = std::make_shared<exec::DynamicShapeInferer>(_ctx, _tensor_reg);
  }
  ret->dynamic_tensor_ctx(dyn_ctx);

  auto &op = _graph.operations().at(ind);
  op.accept(*this);
  assert(_return_fn); // _return_fn must have been generated
  ret->append(std::move(_return_fn));

  for (auto &&ind : (op.getInputs() | ir::Remove::UNDEFINED) + op.getOutputs())
  {
    auto tensor = _tensor_reg->getNativeTensor(ind);
    if (tensor)
    {
      tensor->increase_ref();
    }
  }
  return ret;
}

void KernelGenerator::visit(const ir::operation::AddN &node)
{
  const auto output_index{node.getOutputs().at(0)};

  std::vector<const IPortableTensor *> input_tensors;
  for (const auto &input_idx : node.getInputs())
    input_tensors.emplace_back(_tensor_reg->getPortableTensor(input_idx));

  auto output_tensor = _tensor_reg->getPortableTensor(output_index);

  auto fn = std::make_unique<ops::AddNLayer>();

  fn->configure(std::move(input_tensors), output_tensor);

  _return_fn = std::move(fn);
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
  const auto &param_padding = node.param().padding;
  const auto dilation = node.param().dilation;

  const bool is_cacheable_weights = ker_tensor->is_constant();

  auto fn = std::make_unique<ops::ConvolutionLayer>();

  if (_ctx.at(ifm_index).info().isDynamic() || _ctx.at(ker_index).info().isDynamic())
  {
    fn->configure(ifm_tensor, ker_tensor, bias_tensor, param_padding.type, param_padding.param.left,
                  param_padding.param.right, param_padding.param.top, param_padding.param.bottom,
                  stride.horizontal, stride.vertical, dilation.width_factor, dilation.height_factor,
                  activation, ofm_tensor, is_cacheable_weights);

    _return_fn = std::move(fn);
    return;
  }
  const auto ifm_shape = _ctx.at(ifm_index).shape().asFeature();
  const auto ofm_shape = _ctx.at(ofm_index).shape().asFeature();
  // Kernel format is [depth_out, kernel_height, kernel_width, depth_in].
  const auto &ker_shape = _ctx.at(ker_index).shape();
  const auto ker_height = ker_shape.dim(1);
  const auto ker_width = ker_shape.dim(2);

  const auto padding =
    ir::calculatePadding(param_padding, ifm_shape, ofm_shape, stride, ker_width, ker_height,
                         dilation.width_factor, dilation.height_factor);

  fn->configure(ifm_tensor, ker_tensor, bias_tensor, param_padding.type, padding.left,
                padding.right, padding.top, padding.bottom, stride.horizontal, stride.vertical,
                dilation.width_factor, dilation.height_factor, activation, ofm_tensor,
                is_cacheable_weights);

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
  const auto ifm_shape = _ctx.at(ifm_index).shape().asFeature();
  const auto ofm_shape = _ctx.at(ofm_index).shape().asFeature();
  // Kernel format is [1, kernel_height, kernel_width, depth_out].
  const auto &ker_shape = _ctx.at(ker_index).shape();
  const auto ker_height = ker_shape.dim(1);
  const auto ker_width = ker_shape.dim(2);
  const auto dilation_width = node.param().dilation.width_factor;
  const auto dilation_height = node.param().dilation.height_factor;
  const auto padding = ir::calculatePadding(node.param().padding, ifm_shape, ofm_shape, stride,
                                            ker_width, ker_height, dilation_width, dilation_height);
  const auto multiplier = node.param().multiplier;
  const auto activation = node.param().activation;

  auto ofm_tensor = _tensor_reg->getPortableTensor(ofm_index);
  auto ifm_tensor = _tensor_reg->getPortableTensor(ifm_index);
  auto ker_tensor = _tensor_reg->getPortableTensor(ker_index);
  auto bias_tensor = _tensor_reg->getPortableTensor(bias_index);

  auto fn = std::make_unique<ops::DepthwiseConvolutionLayer>();

  fn->configure(ifm_tensor, ker_tensor, bias_tensor, padding.left, padding.right, padding.top,
                padding.bottom, stride.horizontal, stride.vertical, multiplier, dilation_width,
                dilation_height, activation, ofm_tensor, _external_context);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Concat &node)
{
  const auto ofm_index{node.getOutputs().at(0)};

  const auto rank = _ctx.at(ofm_index).shape().rank();
  const auto axis = ops::getAxis(rank, node.param().axis);

  auto output_tensor = _tensor_reg->getPortableTensor(ofm_index);

  std::vector<const IPortableTensor *> input_tensors;
  for (const auto &ifm_idx : node.getInputs())
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
  // SHAPE input is used for shape inference
  const auto value_index{node.getInputs().at(ir::operation::Fill::Input::VALUE)};

  auto output_tensor = _tensor_reg->getPortableTensor(output_index);
  auto value_tensor = _tensor_reg->getPortableTensor(value_index);

  auto fn = std::make_unique<ops::FillLayer>();

  fn->configure(value_tensor, output_tensor);

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
  const auto weights_format = node.param().weights_format;

  auto output_tensor = _tensor_reg->getPortableTensor(output_index);
  auto input_tensor = _tensor_reg->getPortableTensor(input_index);
  auto weight_tensor = _tensor_reg->getPortableTensor(weight_index);
  auto bias_tensor = bias_index.undefined() ? nullptr : _tensor_reg->getPortableTensor(bias_index);

  auto fn = std::make_unique<ops::FullyConnectedLayer>();

  fn->configure(input_tensor, weight_tensor, bias_tensor, activation, weights_format, output_tensor,
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

  const auto rank = _ctx.at(input_index).shape().rank();
  const auto axis = ops::getAxis(rank, node.param().axis);

  auto fn = std::make_unique<ops::GatherLayer>();

  fn->configure(input_tensor, indices_tensor, output_tensor, axis, _external_context.get());

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
  assert(axis <= static_cast<int>(indices_tensor->getShape().rank()));

  auto fn = std::make_unique<ops::OneHotLayer>();

  fn->configure(indices_tensor, depth_tensor, onvalue_tensor, offvalue_tensor, output_tensor, axis);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Einsum &node)
{
  const auto ofm_index{node.getOutputs().at(0)};

  auto output_tensor = _tensor_reg->getPortableTensor(ofm_index);
  std::vector<const IPortableTensor *> input_tensors;
  for (const auto &ifm_idx : node.getInputs())
    input_tensors.emplace_back(_tensor_reg->getPortableTensor(ifm_idx));

  const auto &equation = node.param().equation;

  auto fn = std::make_unique<ops::EinsumLayer>();

  fn->configure(input_tensors, equation, output_tensor);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Custom &node)
{
  auto fill_op_info = [&](const ir::OperandIndexSequence &opSeq,
                          std::vector<custom::TypeInfo> &types,
                          std::vector<IPortableTensor *> &tensors) {
    for (const auto &idx : opSeq)
    {
      const auto &operand = _ctx.at(idx);
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

  if (node.param().op_type == ir::operation::ElementwiseUnary::Type::QUANTIZE)
  {
    auto fn = std::make_unique<ops::QuantizeLayer>();
    fn->configure(input_tensor, output_tensor);
    _return_fn = std::move(fn);
  }
  else
  {
    auto fn = std::make_unique<ops::ElementwiseUnaryLayer>();
    fn->configure(input_tensor, output_tensor, convertElementwiseUnaryType(node.param().op_type));
    _return_fn = std::move(fn);
  }
}

void KernelGenerator::visit(const ir::operation::ExpandDims &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::ExpandDims::Input::INPUT)};
  // AXIS input is used for output shape inference

  auto output_tensor = _tensor_reg->getPortableTensor(output_index);
  auto input_tensor = _tensor_reg->getPortableTensor(input_index);

  auto fn = std::make_unique<ops::ExpandDimsLayer>();

  fn->configure(input_tensor, output_tensor);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Pack &node)
{
  const auto ofm_index{node.getOutputs().at(0)};

  const auto rank = _ctx.at(ofm_index).shape().rank();
  const auto axis = ops::getAxis(rank, node.param().axis);

  assert(-rank <= axis && axis < rank);

  auto output_tensor = _tensor_reg->getPortableTensor(ofm_index);

  std::vector<const IPortableTensor *> input_tensors;
  for (const auto &ifm_idx : node.getInputs())
    input_tensors.emplace_back(_tensor_reg->getPortableTensor(ifm_idx));

  auto fn = std::make_unique<ops::PackLayer>();

  fn->configure(input_tensors, axis, output_tensor);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Unpack &node)
{
  const auto input_index{node.getInputs().at(0)};

  const auto rank = _ctx.at(input_index).shape().rank();
  const auto axis = ops::getAxis(rank, node.param().axis);

  assert(rank == 0 || (-rank <= axis && axis < rank));

  auto input_tensor = _tensor_reg->getPortableTensor(input_index);

  std::vector<IPortableTensor *> output_tensors;
  for (const auto &output_idx : node.getOutputs())
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

  auto input = _tensor_reg->getPortableTensor(input_index);
  auto pad = _tensor_reg->getPortableTensor(pad_index);
  auto output = _tensor_reg->getPortableTensor(output_index);

  auto fn = std::make_unique<ops::PadLayer>();

  IPortableTensor *value = nullptr;
  if (node.getInputs().size() == 3) // isPadV2
  {
    const auto value_index{node.getInputs().at(ir::operation::Pad::Input::VALUE)};
    value = _tensor_reg->getPortableTensor(value_index);
  }

  fn->configure(input, pad, value, output);
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
  for (const auto &output_idx : node.getOutputs())
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

void KernelGenerator::visit(const ir::operation::ArgMinMax &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::ArgMinMax::INPUT)};
  const auto axis_index{node.getInputs().at(ir::operation::ArgMinMax::AXIS)};

  auto output_tensor = _tensor_reg->getPortableTensor(output_index);
  auto input_tensor = _tensor_reg->getPortableTensor(input_index);
  auto axis_tensor = _tensor_reg->getPortableTensor(axis_index);

  auto fn = std::make_unique<ops::ArgMinMaxLayer>();

  fn->configure(input_tensor, output_tensor, axis_tensor, node.param().is_arg_max);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Pool2D &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::Pool2D::Input::INPUT)};

  const auto kh = node.param().kh;
  const auto kw = node.param().kw;
  const auto stride = node.param().stride;
  const auto ifm_shape = _ctx.at(ifm_index).shape().asFeature();
  const auto ofm_shape = _ctx.at(ofm_index).shape().asFeature();
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

void KernelGenerator::visit(const ir::operation::RmsNorm &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::RmsNorm::Input::INPUT)};
  const auto gamma_index{node.getInputs().at(ir::operation::RmsNorm::Input::GAMMA)};

  auto ofm_tensor = _tensor_reg->getPortableTensor(ofm_index);
  auto ifm_tensor = _tensor_reg->getPortableTensor(ifm_index);
  auto gamma_tensor = _tensor_reg->getPortableTensor(gamma_index);
  auto epsilon = node.param().epsilon;

  auto fn = std::make_unique<ops::RmsNormLayer>();

  fn->configure(ifm_tensor, gamma_tensor, epsilon, ofm_tensor);

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

void KernelGenerator::visit(const ir::operation::DetectionPostProcess &node)
{
  using NMS = ir::operation::DetectionPostProcess;

  ops::DetectionPostProcessLayer::DetectionPostProcessParameters parameters;
  parameters.scales.y = node.param().scale.y_scale;
  parameters.scales.x = node.param().scale.x_scale;
  parameters.scales.w = node.param().scale.w_scale;
  parameters.scales.h = node.param().scale.h_scale;

  parameters.iou_threshold = node.param().iou_threshold;
  parameters.score_threshold = node.param().score_threshold;
  parameters.max_boxes_per_class = node.param().max_boxes_per_class;
  parameters.max_detections = node.param().max_detections;
  parameters.num_classes = node.param().num_classes;
  parameters.center_box_format = node.param().center_size_boxes;
  parameters.max_classes_per_detection = node.param().max_classes_per_detection;

  auto boxes_index = node.getInputs().at(NMS::Input::BOXES);
  auto scores_index = node.getInputs().at(NMS::Input::SCORES);
  auto anchors_index = node.getInputs().at(NMS::Input::INPUT_ANCHORS);

  auto o_classes_index = node.getOutputs().at(NMS::Output::BOX_CLASSES);
  auto o_coords_index = node.getOutputs().at(NMS::Output::BOX_COORDS);
  auto o_scores_index = node.getOutputs().at(NMS::Output::BOX_SCORES);
  auto o_num_selected_index = node.getOutputs().at(NMS::Output::NUM_SELECTED);

  parameters.boxes_descr = _ctx.at(boxes_index).shape().dims();
  parameters.scrores_descr = _ctx.at(scores_index).shape().dims();

  parameters.boxes_input = _tensor_reg->getPortableTensor(boxes_index);
  parameters.scores_input = _tensor_reg->getPortableTensor(scores_index);
  parameters.anchors_input = _tensor_reg->getPortableTensor(anchors_index);

  parameters.box_classes_output = _tensor_reg->getPortableTensor(o_classes_index);
  parameters.box_coords_output = _tensor_reg->getPortableTensor(o_coords_index);
  parameters.box_scores_output = _tensor_reg->getPortableTensor(o_scores_index);
  parameters.num_selections_output = _tensor_reg->getPortableTensor(o_num_selected_index);

  auto fn = std::make_unique<ops::DetectionPostProcessLayer>();
  fn->configure(std::move(parameters));

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

  fn->configure(lhs_tensor, rhs_tensor, adj_x, adj_y, output_tensor, _external_context.get());
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
  for (const auto &ifm_idx : node.getInputs())
    input_tensors.emplace_back(_tensor_reg->getPortableTensor(ifm_idx));

  const auto epsilon = node.param().epsilon;
  const auto is_training = node.param().is_training;
  const auto &data_format = node.param().data_format;

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

void KernelGenerator::visit(const ir::operation::DepthToSpace &node)
{
  const auto input_index{node.getInputs().at(ir::operation::DepthToSpace::Input::INPUT)};
  const auto output_index{node.getOutputs().at(0)};
  auto block_size = node.param().block_size;

  auto input_tensor = _tensor_reg->getPortableTensor(input_index);
  auto output_tensor = _tensor_reg->getPortableTensor(output_index);

  auto fn = std::make_unique<ops::DepthToSpaceLayer>();

  fn->configure(input_tensor, block_size, output_tensor);
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
  for (const auto &output_idx : node.getOutputs())
    out_tensors.emplace_back(_tensor_reg->getPortableTensor(output_idx));

  auto fn = std::make_unique<ops::SplitVLayer>();

  fn->configure(in_tensor, in_size_splits, in_split_dim, num_splits, out_tensors);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::LSTM &node)
{
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
  const auto cell_gate_bias_index{node.getInputs().at(ir::operation::LSTM::Input::CELL_BIAS)};
  const auto output_gate_bias_index{
    node.getInputs().at(ir::operation::LSTM::Input::OUTPUT_GATE_BIAS)};
  const auto projection_weights_index{
    node.getInputs().at(ir::operation::LSTM::Input::PROJECTION_WEIGHTS)}; // optional
  const auto projection_bias_index{
    node.getInputs().at(ir::operation::LSTM::Input::PROJECTION_BIAS)}; // optional
  const auto output_state_in_index{
    node.getInputs().at(ir::operation::LSTM::Input::OUTPUT_STATE_IN)};
  const auto cell_state_in_index{node.getInputs().at(ir::operation::LSTM::Input::CELL_STATE_IN)};
  const auto time_major = node.param().time_major;

  // NOTE The input_to_input_weights and the recurrent_to_input_weights do not exist in CIFG.
  // has_input_to_input_weights && has_recurrent_to_input_weights: no CIFG
  // !(has_input_to_input_weights && has_recurrent_to_input_weights): CIFG
  // NOTE The cell_to_input_weights does not exist in non-peephole although regular LSTM(non-CIFG).
  bool has_input_to_input_weights = _ctx.exist(input_to_input_weights_index) &&
                                    (_ctx.at(input_to_input_weights_index).shape().dim(0) != 0 &&
                                     _ctx.at(input_to_input_weights_index).shape().dim(1) != 0);
  bool has_recurrent_to_input_weights =
    _ctx.exist(recurrent_to_input_weights_index) &&
    (_ctx.at(recurrent_to_input_weights_index).shape().dim(0) != 0 &&
     _ctx.at(recurrent_to_input_weights_index).shape().dim(1) != 0);

  // NOTE The cell_to_forget_weights and the cell_to_output_weights exist in peephole.
  // But the cell_to_input_weights does not exist in regular CIFG although peephole.
  // has_cell_to_forget_weights && has_cell_to_output_weights: peephole
  // !(has_cell_to_forget_weights && has_cell_to_output_weights): no peephole
  bool has_cell_to_forget_weights = _ctx.exist(cell_to_forget_weights_index) &&
                                    _ctx.at(cell_to_forget_weights_index).shape().dim(0) != 0;
  bool has_cell_to_output_weights = _ctx.exist(cell_to_output_weights_index) &&
                                    _ctx.at(cell_to_output_weights_index).shape().dim(0) != 0;

  bool has_input_gate_bias =
    _ctx.exist(input_gate_bias_index) && _ctx.at(input_gate_bias_index).shape().dim(0);

  bool has_projection_weights = _ctx.exist(projection_weights_index) &&
                                (_ctx.at(projection_weights_index).shape().dim(0) != 0 &&
                                 _ctx.at(projection_weights_index).shape().dim(1) != 0);
  bool has_projection_bias =
    _ctx.exist(projection_bias_index) && _ctx.at(projection_bias_index).shape().dim(0);

  auto scratch_buffer_tensor = _ctx.exist(scratch_buffer_index)
                                 ? _tensor_reg->getPortableTensor(scratch_buffer_index)
                                 : nullptr; // optional
  auto output_state_out_tensor = _ctx.exist(output_state_out_index)
                                   ? _tensor_reg->getPortableTensor(output_state_out_index)
                                   : nullptr; // optional
  auto cell_state_out_tensor = _ctx.exist(cell_state_out_index)
                                 ? _tensor_reg->getPortableTensor(cell_state_out_index)
                                 : nullptr; // optional
  auto output_tensor = _tensor_reg->getPortableTensor(output_index);

  auto input_tensor = _tensor_reg->getPortableTensor(input_index);

  auto input_to_input_weights_tensor =
    has_input_to_input_weights ? _tensor_reg->getPortableTensor(input_to_input_weights_index)
                               : nullptr; // optional
  auto input_to_forget_weights_tensor =
    _tensor_reg->getPortableTensor(input_to_forget_weights_index);
  auto input_to_cell_weights_tensor = _tensor_reg->getPortableTensor(input_to_cell_weights_index);
  auto input_to_output_weights_tensor =
    _tensor_reg->getPortableTensor(input_to_output_weights_index);
  auto recurrent_to_input_weights_tensor =
    has_recurrent_to_input_weights
      ? _tensor_reg->getPortableTensor(recurrent_to_input_weights_index)
      : nullptr; // optional
  auto recurrent_to_forget_weights_tensor =
    _tensor_reg->getPortableTensor(recurrent_to_forget_weights_index);
  auto recurrent_to_cell_weights_tensor =
    _tensor_reg->getPortableTensor(recurrent_to_cell_weights_index);
  auto recurrent_to_output_weights_tensor =
    _tensor_reg->getPortableTensor(recurrent_to_output_weights_index);

  auto cell_to_input_weights_tensor = _tensor_reg->getPortableTensor(cell_to_input_weights_index);
  auto cell_to_forget_weights_tensor =
    has_cell_to_forget_weights ? _tensor_reg->getPortableTensor(cell_to_forget_weights_index)
                               : nullptr; // optional
  auto cell_to_output_weights_tensor =
    has_cell_to_output_weights ? _tensor_reg->getPortableTensor(cell_to_output_weights_index)
                               : nullptr; // optional

  auto input_gate_bias_tensor =
    has_input_gate_bias ? _tensor_reg->getPortableTensor(input_gate_bias_index) : nullptr;
  auto forget_gate_bias_tensor = _tensor_reg->getPortableTensor(forget_gate_bias_index);
  auto cell_gate_bias_tensor = _tensor_reg->getPortableTensor(cell_gate_bias_index);
  auto output_gate_bias_tensor = _tensor_reg->getPortableTensor(output_gate_bias_index);
  auto output_state_in_tensor = _tensor_reg->getPortableTensor(output_state_in_index);
  auto cell_state_in_tensor = _tensor_reg->getPortableTensor(cell_state_in_index);

  auto projection_weights_tensor = has_projection_weights
                                     ? _tensor_reg->getPortableTensor(projection_weights_index)
                                     : nullptr; // optional
  auto projection_bias_tensor = has_projection_bias
                                  ? _tensor_reg->getPortableTensor(projection_bias_index)
                                  : nullptr; // optional

  IPortableTensor *input_layer_norm_weights_tensor = nullptr;
  IPortableTensor *forget_layer_norm_weights_tensor = nullptr;
  IPortableTensor *cell_layer_norm_weights_tensor = nullptr;
  IPortableTensor *output_layer_norm_weights_tensor = nullptr;
  if (node.getInputs().size() == 24)
  {
    const auto input_layer_norm_weights_index{
      node.getInputs().at(ir::operation::LSTM::Input::INPUT_LAYER_NORMALIZATION_WEIGHTS)};
    const auto forget_layer_norm_weights_index{
      node.getInputs().at(ir::operation::LSTM::Input::FORGET_LAYER_NORMALIZATION_WEIGHTS)};
    const auto cell_layer_norm_weights_index{
      node.getInputs().at(ir::operation::LSTM::Input::CELL_LAYER_NORMALIZATION_WEIGHTS)};
    const auto output_layer_norm_weights_index{
      node.getInputs().at(ir::operation::LSTM::Input::OUTPUT_LAYER_NORMALIZATION_WEIGHTS)};

    input_layer_norm_weights_tensor =
      _tensor_reg->getPortableTensor(input_layer_norm_weights_index);
    forget_layer_norm_weights_tensor =
      _tensor_reg->getPortableTensor(forget_layer_norm_weights_index);
    cell_layer_norm_weights_tensor = _tensor_reg->getPortableTensor(cell_layer_norm_weights_index);
    output_layer_norm_weights_tensor =
      _tensor_reg->getPortableTensor(output_layer_norm_weights_index);
  }

  auto fn = std::make_unique<ops::LSTMLayer>();

  fn->configure(
    input_tensor, input_to_input_weights_tensor, input_to_forget_weights_tensor,
    input_to_cell_weights_tensor, input_to_output_weights_tensor, recurrent_to_input_weights_tensor,
    recurrent_to_forget_weights_tensor, recurrent_to_cell_weights_tensor,
    recurrent_to_output_weights_tensor, cell_to_input_weights_tensor, cell_to_forget_weights_tensor,
    cell_to_output_weights_tensor, input_layer_norm_weights_tensor,
    forget_layer_norm_weights_tensor, cell_layer_norm_weights_tensor,
    output_layer_norm_weights_tensor,
    /*aux_input=*/nullptr,
    /*aux_input_to_input_weights=*/nullptr,
    /*aux_input_to_forget_weights=*/nullptr,
    /*aux_input_to_cell_weights=*/nullptr,
    /*aux_input_to_output_weights=*/nullptr, input_gate_bias_tensor, forget_gate_bias_tensor,
    cell_gate_bias_tensor, output_gate_bias_tensor, projection_weights_tensor,
    projection_bias_tensor, output_state_in_tensor, cell_state_in_tensor, node.param(),
    /*forward_sequence=*/true, time_major,
    /*output_offset=*/0, scratch_buffer_tensor, output_state_out_tensor, cell_state_out_tensor,
    output_tensor,
    !_ctx.at(output_state_in_index).info().isVariable() /* means empty buffer on frontend now */,
    !_ctx.at(cell_state_in_index).info().isVariable());

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::RoPE &node)
{
  const auto input_index{node.getInputs().at(ir::operation::RoPE::Input::INPUT)};
  const auto sin_table{node.getInputs().at(ir::operation::RoPE::Input::SIN_TABLE)};
  const auto cos_table{node.getInputs().at(ir::operation::RoPE::Input::COS_TABLE)};
  const auto output_index{node.getOutputs().at(ir::operation::RoPE::Output::OUTPUT)};

  auto mode = ops::getRoPEMode(node.param().mode);

  auto input_tensor = _tensor_reg->getPortableTensor(input_index);
  auto sin_tensor = _tensor_reg->getPortableTensor(sin_table);
  auto cos_tensor = _tensor_reg->getPortableTensor(cos_table);
  auto output_tensor = _tensor_reg->getPortableTensor(output_index);

  auto fn = std::make_unique<ops::RoPELayer>();

  fn->configure(input_tensor, sin_tensor, cos_tensor, mode, output_tensor);
  _return_fn = std::move(fn);
}

} // namespace cpu
} // namespace backend
} // namespace onert
