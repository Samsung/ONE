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

#include "ClFunction.h"
#include "TensorManager.h"

#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/elementwise.h"
#include "tensorflow/lite/delegates/gpu/common/selectors/convolution_selector.h"
#include "tensorflow/lite/delegates/gpu/common/selectors/dw_convolution_selector.h"
#include "tensorflow/lite/delegates/gpu/common/selectors/simple_selectors.h"

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

void KernelGenerator::addClNode(const std::vector<ir::OperandIndex> &inputs,
                                const std::vector<ir::OperandIndex> &outputs,
                                std::unique_ptr<tflite::gpu::GPUOperation> gpu_op)
{
  tflite::gpu::cl::CLNode cl_node;
  cl_node.cl_operation.Init(std::move(gpu_op));
  cl_node.inputs.resize(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i)
  {
    cl_node.inputs[i] = inputs[i].value();
  }
  cl_node.outputs.resize(outputs.size());
  for (size_t i = 0; i < outputs.size(); ++i)
  {
    cl_node.outputs[i] = outputs[i].value();
  }
  _nodes.push_back(std::move(cl_node));
  _operation_indexes.push_back(_operation_index);
  return;
}

void KernelGenerator::get_operation(FunctionMap &Functions)
{
  size_t size = _nodes.size();
  size_t i = 0;
  for (auto &&it : Functions)
  {
    auto index = it.first;
    auto node_index = _operation_indexes[i];
    while (index == node_index)
    {
      auto &fn_seq = it.second;
      auto &node = _nodes[i++];
      for (size_t j = 0; j < node.inputs.size(); ++j)
      {
        uint32_t idx = node.inputs[j];
        node.cl_operation.GetGpuOperation().SetSrc(
          _tensor_reg->getClTensor(ir::OperandIndex{idx})->handle(), j);
      }
      for (size_t j = 0; j < node.outputs.size(); ++j)
      {
        uint32_t idx = node.outputs[j];
        node.cl_operation.GetGpuOperation().SetDst(
          _tensor_reg->getClTensor(ir::OperandIndex{idx})->handle(), j);
      }
      fn_seq->iterate([&](exec::IFunction &ifunc) {
        static_cast<ClFunction &>(ifunc).add_operation(&node.cl_operation);
      });
      if (i == size)
      {
        break;
      }
      node_index = _operation_indexes[i];
    }
    if (i == size)
    {
      break;
    }
  }
}

absl::Status KernelGenerator::readConstTensor(const ir::OperandIndex &index,
                                              tflite::gpu::TensorOrScalar *param)
{
  const auto shape = _ctx.at(index).shape();
  if (shape.rank() == 0 && shape.num_elements() == 1)
  {
    tflite::gpu::Tensor<tflite::gpu::Scalar, tflite::gpu::DataType::FLOAT32> tensor;
    tensor.shape.v = 1;
    tensor.data.resize(1);
    std::memcpy(&tensor.data[0], _ctx.at(index).data()->base(), _ctx.at(index).operandSize());
    *param = tensor.data[0];
  }
  else
  {
    if (CheckIfLinearConvertible(&shape))
    {
      tflite::gpu::Tensor<tflite::gpu::Linear, tflite::gpu::DataType::FLOAT32> tensor;
      tensor.shape.v = shape.dim(shape.rank() - 1);
      tensor.data.resize(shape.num_elements());
      std::memcpy(&tensor.data[0], _ctx.at(index).data()->base(), _ctx.at(index).operandSize());
      *param = std::move(tensor);
    }
    else
    {
      tflite::gpu::Tensor<tflite::gpu::HWC, tflite::gpu::DataType::FLOAT32> tensor;
      if (shape.rank() == 3)
      {
        tensor.shape.h = shape.dim(0);
        tensor.shape.w = shape.dim(1);
        tensor.shape.c = shape.dim(2);
      }
      else if (shape.rank() == 4)
      {
        if (shape.dim(0) != 1)
        {
          return absl::UnimplementedError("Batch size is not equal to 1.");
        }
        tensor.shape.h = shape.dim(1);
        tensor.shape.w = shape.dim(2);
        tensor.shape.c = shape.dim(3);
      }
      else
      {
        return absl::InvalidArgumentError(
          "Expected a 3D tensor of shape HxWxC or a 4D tensor of shape 1xHxWxC.");
      }
      tensor.data.resize(shape.num_elements());
      std::memcpy(&tensor.data[0], _ctx.at(index).data()->base(), _ctx.at(index).operandSize());
      *param = std::move(tensor);
    }
  }
  return absl::OkStatus();
}

absl::Status KernelGenerator::readConstTensor(
  const ir::OperandIndex &index,
  absl::variant<tflite::gpu::Tensor<tflite::gpu::Linear, tflite::gpu::DataType::FLOAT32>,
                tflite::gpu::Tensor<tflite::gpu::HWC, tflite::gpu::DataType::FLOAT32>> *alpha)
{
  const auto &shape = _ctx.at(index).shape();
  if (CheckIfLinearConvertible(&shape))
  {
    tflite::gpu::Tensor<tflite::gpu::Linear, tflite::gpu::DataType::FLOAT32> tensor;
    tensor.shape.v = shape.dim(shape.rank() - 1);
    tensor.data.resize(shape.num_elements());
    std::memcpy(&tensor.data[0], _ctx.at(index).data()->base(), _ctx.at(index).operandSize());
    *alpha = std::move(tensor);
  }
  else
  {
    tflite::gpu::Tensor<tflite::gpu::HWC, tflite::gpu::DataType::FLOAT32> tensor;
    if (shape.rank() == 3)
    {
      tensor.shape.h = shape.dim(0);
      tensor.shape.w = shape.dim(1);
      tensor.shape.c = shape.dim(2);
    }
    else if (shape.rank() == 4)
    {
      if (shape.dim(0) != 1)
      {
        return absl::UnimplementedError("Batch size is not equal to 1.");
      }
      tensor.shape.h = shape.dim(1);
      tensor.shape.w = shape.dim(2);
      tensor.shape.c = shape.dim(3);
    }
    else
    {
      return absl::InvalidArgumentError(
        "Expected a 3D tensor of shape HxWxC or a 4D tensor of shape 1xHxWxC.");
    }
    tensor.data.resize(shape.num_elements());
    std::memcpy(&tensor.data[0], _ctx.at(index).data()->base(), _ctx.at(index).operandSize());
    *alpha = std::move(tensor);
  }
  return absl::OkStatus();
}

KernelGenerator::KernelGenerator(
  const ir::Graph &graph, const std::shared_ptr<TensorBuilder> &tensor_builder,
  const std::shared_ptr<TensorRegistry> &tensor_reg,
  const std::shared_ptr<tflite::gpu::cl::CreationContext> &creation_context)
  : basic::KernelGeneratorBase{graph}, _ctx(graph.operands()),
    _operations_ctx(graph.operations()), _current_layout{graph.layout()},
    _tensor_builder(tensor_builder), _tensor_reg(tensor_reg), _creation_context(creation_context)
{
}

std::unique_ptr<exec::FunctionSequence> KernelGenerator::generate(ir::OperationIndex ind)
{
  auto fn_seq = std::make_unique<exec::FunctionSequence>();
  fn_seq->enableDynamicShapeInferer(false);
  _operation_index = ind;
  const auto &op = _graph.operations().at(ind);
  op.accept(*this);
  fn_seq->append(releaseFunction());
  return fn_seq;
}

void KernelGenerator::visit(const ir::operation::BinaryArithmetic &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto lhs_index{node.getInputs().at(ir::operation::BinaryArithmetic::Input::LHS)};
  const auto rhs_index{node.getInputs().at(ir::operation::BinaryArithmetic::Input::RHS)};

  tflite::gpu::OperationDef op_def;
  op_def.precision = tflite::gpu::CalculationsPrecision::F32;

  const bool lhs_const = _ctx.at(lhs_index).isConstant();
  const bool rhs_const = _ctx.at(rhs_index).isConstant();

  if (lhs_const && rhs_const)
  {
    throw std::runtime_error("No runtime input tensors for " + node.name());
  }

  auto fn = std::make_unique<ClFunction>(_creation_context);
  std::unique_ptr<tflite::gpu::GPUOperation> gpu_op;

  tflite::gpu::OperationType op_type = convertArithmeticType(node.param().arithmetic_type);

  if (!lhs_const && !rhs_const)
  {
    auto lhs_shape = _tensor_reg->getClTensor(lhs_index)->get_info()._shape;
    auto rhs_shape = _tensor_reg->getClTensor(rhs_index)->get_info()._shape;

    bool swap =
      (op_type == tflite::gpu::OperationType::MUL) &&
      (lhs_shape.h <= rhs_shape.h && lhs_shape.w <= rhs_shape.w && lhs_shape.c <= rhs_shape.c);

    auto first_index = swap ? rhs_index : lhs_index;
    auto second_index = swap ? lhs_index : rhs_index;

    op_def.src_tensors.push_back(_tensor_reg->getClTensor(first_index)->get_info()._desc);
    op_def.src_tensors.push_back(_tensor_reg->getClTensor(second_index)->get_info()._desc);
    op_def.dst_tensors.push_back(_tensor_reg->getClTensor(ofm_index)->get_info()._desc);

    auto second_shape = _tensor_reg->getClTensor(second_index)->get_info()._shape;

    tflite::gpu::GPUOperation operation = CreateElementwiseTwoInput(op_def, op_type, second_shape);
    gpu_op = std::make_unique<tflite::gpu::GPUOperation>(std::move(operation));

    addClNode({first_index, second_index}, {ofm_index}, std::move(gpu_op));
  }
  else
  {
    auto non_const_index = rhs_const ? lhs_index : rhs_index;
    auto const_index = rhs_const ? rhs_index : lhs_index;

    op_def.dst_tensors.push_back(_tensor_reg->getClTensor(ofm_index)->get_info()._desc);
    op_def.src_tensors.push_back(_tensor_reg->getClTensor(non_const_index)->get_info()._desc);

    tflite::gpu::ElementwiseAttributes attr;

    if (!readConstTensor(const_index, &attr.param).ok())
    {
      throw std::runtime_error("BinaryArithmetic unsupported constant tensor");
    }

    tflite::gpu::GPUOperation operation =
      CreateElementwise(_creation_context->GetGpuInfo(), op_def, op_type, attr);
    gpu_op = absl::make_unique<tflite::gpu::GPUOperation>(std::move(operation));

    addClNode({non_const_index}, {ofm_index}, std::move(gpu_op));
  }
  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Conv2D &node)
{
  auto output{node.getOutputs().at(0)};

  auto input{node.getInputs().at(ir::operation::Conv2D::INPUT)};
  auto kernel{node.getInputs().at(ir::operation::Conv2D::KERNEL)};
  auto bias{node.getInputs().at(ir::operation::Conv2D::BIAS)};

  const auto &param = node.param();

  tflite::gpu::OperationDef op_def;
  op_def.precision = tflite::gpu::CalculationsPrecision::F32;

  op_def.src_tensors.push_back(_tensor_reg->getClTensor(input)->get_info()._desc);

  auto input_shape = _tensor_reg->getClTensor(input)->get_info()._shape;
  auto kernel_shape = _tensor_reg->getClTensor(kernel)->get_info()._shape;
  auto output_shape = _tensor_reg->getClTensor(output)->get_info()._shape;
  auto bias_shape = _tensor_reg->getClTensor(bias)->get_info()._shape;

  op_def.dst_tensors.push_back(_tensor_reg->getClTensor(output)->get_info()._desc);

  tflite::gpu::ModelHints hints;
  std::unique_ptr<tflite::gpu::GPUOperation>
    gpu_op; // = InitSingleOpSubgraph(inputs, outputs, gpu_subgraph);

  auto kernel_tensor = _tensor_reg->getClTensor(kernel);
  auto bias_tensor = _tensor_reg->getClTensor(bias);

  tflite::gpu::Convolution2DAttributes attr;
  attr.strides = ToHW(param.stride.vertical, param.stride.horizontal);
  attr.dilations =
    tflite::gpu::HW(std::max(static_cast<u_int32_t>(1), param.dilation.height_factor),
                    std::max(static_cast<u_int32_t>(1), param.dilation.width_factor));

  bool is_weight = (_ctx.at(kernel).isConstant() ? true : false);

  if (is_weight)
  {
    attr.weights.id = kernel.value();
    attr.weights.shape.o = kernel_shape.b;
    attr.weights.shape.h = kernel_shape.h;
    attr.weights.shape.w = kernel_shape.w;
    attr.weights.shape.i = kernel_shape.c;
    attr.weights.data.resize(kernel_shape.DimensionsProduct());
    memcpy(attr.weights.data.data(), _ctx.at(kernel).data()->base(), kernel_tensor->total_size());
  }

  attr.bias.id = bias.value();
  // TODO Modify
  attr.bias.shape.v = bias_shape.b != 1 ? bias_shape.b : bias_shape.c;
  attr.bias.data.resize(bias_shape.DimensionsProduct());
  memcpy(attr.bias.data.data(), _ctx.at(bias).data()->base(), bias_tensor->total_size());

  UpdatePadding(param.padding.type, input_shape, &attr);

  gpu_op = SelectConvolution(attr, output_shape, _creation_context->GetGpuInfo(), op_def, hints);

  tflite::gpu::cl::CLNode cl_node;
  cl_node.inputs.resize(1);
  cl_node.inputs[0] = input.value();
  cl_node.outputs.resize(1);

  auto fn = std::make_unique<ClFunction>(_creation_context);

  const auto activation = node.param().activation;

  switch (activation)
  {
    case ir::Activation::NONE:
    {
      addClNode({input}, {output}, std::move(gpu_op));
      break;
    }
    case ir::Activation::RELU:
    case ir::Activation::RELU6:
    {
      std::unique_ptr<tflite::gpu::GPUOperation> gpu_op_1;
      tflite::gpu::OperationDef op_def_1;
      const auto &shape = _ctx.at(output).shape();
      auto new_ind = _tensor_reg->addNewClTensor(shape);

      addClNode({input}, {new_ind}, std::move(gpu_op));

      op_def_1.precision = tflite::gpu::CalculationsPrecision::F32;
      op_def_1.src_tensors.push_back(_tensor_reg->getClTensor(output)->get_info()._desc);
      op_def_1.dst_tensors.push_back(_tensor_reg->getClTensor(output)->get_info()._desc);

      tflite::gpu::ReLUAttributes attr_1;
      if (activation == ir::Activation::RELU6)
      {
        attr_1.clip = 6;
      }
      else
      {
        attr_1.clip = 0;
      }
      attr_1.alpha = 0;
      gpu_op_1 = SelectReLU(attr_1, op_def_1);

      addClNode({new_ind}, {output}, std::move(gpu_op_1));
      break;
    }
    default:
    {
      throw std::runtime_error("gpu_cl KernelGenerator : Not supported Conv2D activiation");
    }
  }
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
  const auto dilation = node.param().dilation;
  const auto &padding = node.param().padding;

  const auto multiplier = node.param().multiplier;

  bool is_weight = (_ctx.at(ker_index).isConstant() ? true : false);
  tflite::gpu::OperationDef op_def;
  op_def.precision = tflite::gpu::CalculationsPrecision::F32;

  op_def.src_tensors.push_back(_tensor_reg->getClTensor(ifm_index)->get_info()._desc);
  auto input_shape = _tensor_reg->getClTensor(ifm_index)->get_info()._shape;

  auto ker_shape = _tensor_reg->getClTensor(ker_index)->get_info()._shape;

  op_def.dst_tensors.push_back(_tensor_reg->getClTensor(ofm_index)->get_info()._desc);
  auto out_shape = _tensor_reg->getClTensor(ofm_index)->get_info()._shape;
  auto bias_shape = _tensor_reg->getClTensor(bias_index)->get_info()._shape;

  tflite::gpu::DepthwiseConvolution2DAttributes attr;
  attr.strides = ToHW(stride.vertical, stride.horizontal);
  attr.dilations = tflite::gpu::HW(std::max(static_cast<u_int32_t>(1), dilation.height_factor),
                                   std::max(static_cast<u_int32_t>(1), dilation.width_factor));

  if (is_weight)
  {
    attr.weights.id = ker_index.value();
    attr.weights.shape.o = ker_shape.b;
    attr.weights.shape.h = ker_shape.h;
    attr.weights.shape.w = ker_shape.w;
    attr.weights.shape.i = ker_shape.c;
    attr.weights.data.resize(ker_shape.DimensionsProduct());
    memcpy(attr.weights.data.data(), _ctx.at(ker_index).data()->base(),
           _ctx.at(ker_index).operandSize());
  }
  attr.bias.id = bias_index.value();
  attr.bias.shape.v = bias_shape.b != 1 ? bias_shape.b : bias_shape.c;
  attr.bias.data.resize(bias_shape.DimensionsProduct());
  memcpy(attr.bias.data.data(), _ctx.at(bias_index).data()->base(),
         _ctx.at(bias_index).operandSize());
  UpdatePadding(padding.type, input_shape, &attr);

  if (multiplier != 1)
  {
    const int input_depth = input_shape.c;
    const int filter_height = ker_shape.h;
    const int filter_width = ker_shape.w;
    const int output_depth = out_shape.c;

    tflite::gpu::Tensor<tflite::gpu::OHWI, tflite::gpu::DataType::FLOAT32> weights;
    weights.id = attr.weights.id;
    weights.shape = tflite::gpu::OHWI(output_depth, filter_height, filter_width, input_depth);
    weights.data.resize(weights.shape.DimensionsProduct());
    float *dst = &weights.data[0];
    for (int j = 0; j < output_depth; ++j)
    {
      const float *src = attr.weights.data.data() + j;
      for (int i = 0; i < filter_height * filter_width; ++i)
      {
        *dst = *src;
        dst++;
        src += output_depth;
      }
    }
    attr.weights = std::move(weights);
  }

  auto fn = std::make_unique<ClFunction>(_creation_context);
  std::unique_ptr<tflite::gpu::GPUOperation> gpu_op;

  if (is_weight)
  {
    gpu_op = SelectDWConvolution(attr, _creation_context->GetGpuInfo(), op_def);
  }
  else
  {
    if (ker_shape.b != 1)
    {
      throw std::runtime_error(
        "No support of depthwise runtime weights with channel multiplier != 1");
    }
    gpu_op = SelectDWConvolutionDynamicWeights(attr, _creation_context->GetGpuInfo(), op_def);
  }

  const auto activation = node.param().activation;

  switch (activation)
  {
    case ir::Activation::NONE:
    {
      addClNode({ifm_index}, {ofm_index}, std::move(gpu_op));
      break;
    }
    case ir::Activation::RELU:
    case ir::Activation::RELU6:
    {
      std::unique_ptr<tflite::gpu::GPUOperation> gpu_op_1;
      tflite::gpu::OperationDef op_def_1;
      const auto shape = _ctx.at(ofm_index).shape();
      auto new_ind = _tensor_reg->addNewClTensor(shape);

      addClNode({ifm_index}, {new_ind}, std::move(gpu_op));

      op_def_1.precision = tflite::gpu::CalculationsPrecision::F32;

      op_def_1.src_tensors.push_back(_tensor_reg->getClTensor(ofm_index)->get_info()._desc);
      op_def_1.dst_tensors.push_back(_tensor_reg->getClTensor(ofm_index)->get_info()._desc);

      tflite::gpu::ReLUAttributes attr_1;
      if (activation == ir::Activation::RELU6)
      {
        attr_1.clip = 6;
      }
      else
      {
        attr_1.clip = 0;
      }
      attr_1.alpha = 0;
      gpu_op_1 = SelectReLU(attr_1, op_def_1);

      addClNode({new_ind}, {ofm_index}, std::move(gpu_op_1));
      break;
    }
    default:
    {
      throw std::runtime_error("gpu_cl KernelGenerator : Not supported DepthwiseConv2D acvivation");
    }
  }

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::ElementwiseActivation &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::ElementwiseActivation::Input::INPUT)};

  tflite::gpu::OperationDef op_def;
  op_def.precision = tflite::gpu::CalculationsPrecision::F32;

  op_def.dst_tensors.push_back(_tensor_reg->getClTensor(output_index)->get_info()._desc);
  op_def.src_tensors.push_back(_tensor_reg->getClTensor(input_index)->get_info()._desc);

  std::unique_ptr<tflite::gpu::GPUOperation> gpu_op;
  auto fn = std::make_unique<ClFunction>(_creation_context);
  switch (node.param().op_type)
  {
    case ir::operation::ElementwiseActivation::Type::LEAKY_RELU:
    case ir::operation::ElementwiseActivation::Type::RELU:
    {
      tflite::gpu::ReLUAttributes attr;
      if (ir::operation::ElementwiseActivation::Type::LEAKY_RELU == node.param().op_type)
      {
        attr.alpha = node.param().alpha;
        attr.clip = 0;
      }
      else
      {
        attr.alpha = node.param().beta;
        attr.clip = node.param().alpha;
      }
      gpu_op = SelectReLU(attr, op_def);
      break;
    }
    case ir::operation::ElementwiseActivation::Type::LOGISTIC:
    {
      if (_ctx.at(input_index).typeInfo().type() != ir::DataType::FLOAT32)
      {
        throw std::runtime_error{"Unsupported data type of LOGISTIC"};
      }
      tflite::gpu::GPUOperation operation =
        CreateElementwiseOneInput(_creation_context->GetGpuInfo(), op_def,
                                  convertElementwiseActivationType(node.param().op_type));
      gpu_op = std::make_unique<tflite::gpu::GPUOperation>(std::move(operation));
      break;
    }
    case ir::operation::ElementwiseActivation::Type::TANH:
    {
      tflite::gpu::GPUOperation operation = CreateElementwiseOneInput(
        _creation_context->GetGpuInfo(), op_def, tflite::gpu::OperationType::TANH);
      gpu_op = std::make_unique<tflite::gpu::GPUOperation>(std::move(operation));
      break;
    }
    default:
      throw std::runtime_error(
        "gpu_cl KernelGenerator : Not supported operation on ElementwiseActivation");
  }
  addClNode({input_index}, {output_index}, std::move(gpu_op));
  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Pool2D &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Pool2D::Input::INPUT)};

  tflite::gpu::OperationDef op_def;
  op_def.precision = tflite::gpu::CalculationsPrecision::F32;

  op_def.src_tensors.push_back(_tensor_reg->getClTensor(input_index)->get_info()._desc);
  auto input_shape = _tensor_reg->getClTensor(input_index)->get_info()._shape;

  op_def.dst_tensors.push_back(_tensor_reg->getClTensor(output_index)->get_info()._desc);

  const auto kh = node.param().kh;
  const auto kw = node.param().kw;
  const auto stride = node.param().stride;
  const auto op_type = convertPoolType(node.param().op_type);

  tflite::gpu::Pooling2DAttributes attributes;
  attributes.type = op_type;
  attributes.kernel = tflite::gpu::HW(kh > 0 ? kh : 1, kw > 0 ? kw : 1);
  attributes.strides = tflite::gpu::HW(stride.vertical > 0 ? stride.vertical : 1,
                                       stride.horizontal > 0 ? stride.horizontal : 1);

  if (node.param().padding.type == ir::PaddingType::SAME)
  {
    attributes.padding = CalculateSamePadding(input_shape, attributes);
  }
  else
  {
    attributes.padding.prepended = tflite::gpu::HW(0, 0);
    attributes.padding.appended = tflite::gpu::HW(0, 0);
  }

  auto fn = std::make_unique<ClFunction>(_creation_context);
  std::unique_ptr<tflite::gpu::GPUOperation> gpu_op;
  gpu_op = SelectPooling(attributes, op_def);

  addClNode({input_index}, {output_index}, std::move(gpu_op));
  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Reshape &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Reshape::Input::INPUT)};

  tflite::gpu::OperationDef op_def;
  op_def.precision = tflite::gpu::CalculationsPrecision::F32;

  op_def.src_tensors.push_back(_tensor_reg->getClTensor(input_index)->get_info()._desc);
  auto input_shape = _tensor_reg->getClTensor(input_index)->get_info()._shape;

  op_def.dst_tensors.push_back(_tensor_reg->getClTensor(output_index)->get_info()._desc);
  auto output_shape = _tensor_reg->getClTensor(output_index)->get_info()._shape;

  tflite::gpu::ReshapeAttributes attr;
  attr.new_shape = output_shape;

  auto fn = std::make_unique<ClFunction>(_creation_context);
  std::unique_ptr<tflite::gpu::GPUOperation> gpu_op;
  const int src_channels = input_shape.c;
  SelectReshape(src_channels, attr.new_shape.c, op_def, &gpu_op);

  addClNode({input_index}, {output_index}, std::move(gpu_op));
  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Softmax &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Softmax::Input::INPUT)};

  const auto beta = node.param().beta;

  if (beta != 1.0)
  {
    throw std::runtime_error("Softmax.beta != 1 is not supported in gpu_cl");
  }

  tflite::gpu::OperationDef op_def;
  op_def.precision = tflite::gpu::CalculationsPrecision::F32;

  op_def.dst_tensors.push_back(_tensor_reg->getClTensor(output_index)->get_info()._desc);

  op_def.src_tensors.push_back(_tensor_reg->getClTensor(input_index)->get_info()._desc);
  auto input_shape = _tensor_reg->getClTensor(input_index)->get_info()._shape;

  auto fn = std::make_unique<ClFunction>(_creation_context);

  std::unique_ptr<tflite::gpu::GPUOperation> gpu_op;
  SelectSoftmax(input_shape, op_def, &gpu_op);

  addClNode({input_index}, {output_index}, std::move(gpu_op));
  _return_fn = std::move(fn);
}

} // namespace gpu_cl
} // namespace backend
} // namespace onert
