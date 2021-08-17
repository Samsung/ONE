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

#include "open_cl/selectors/ConvolutionSelector.h"
#include "open_cl/selectors/DwConvolutionSelector.h"
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

HW ToHW(int32_t h, int32_t w) { return HW(h > 0 ? h : 1, w > 0 ? w : 1); }

template <typename AttrT>
void UpdatePadding(const ir::PaddingType type, const BHWC &input_shape, AttrT *attr)
{
  if (type == ir::PaddingType::SAME)
  {
    attr->padding = CalculateSamePadding(input_shape, *attr);
  }
  else
  {
    attr->padding.prepended = HW(0, 0);
    attr->padding.appended = HW(0, 0);
  }
}

gpu_cl::PoolingType convertPoolType(ir::operation::Pool2D::PoolType type_ir)
{
  switch (type_ir)
  {
    case ir::operation::Pool2D::PoolType::AVG:
      return gpu_cl::PoolingType::AVERAGE;
    case ir::operation::Pool2D::PoolType::MAX:
      return gpu_cl::PoolingType::MAX;
    default:
      throw std::runtime_error("gpu_Cl KernelGenerator : Not supported operation yet");
  }
}

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

void KernelGenerator::visit(const ir::operation::Conv2D &node)
{
  auto output{node.getOutputs().at(0)};

  auto input{node.getInputs().at(ir::operation::Conv2D::INPUT)};
  auto kernel{node.getInputs().at(ir::operation::Conv2D::KERNEL)};
  auto bias{node.getInputs().at(ir::operation::Conv2D::BIAS)};

  const auto param = node.param();

  OperationDef op_def;
  op_def.precision = CalculationsPrecision::F32;

  op_def.src_tensors.push_back(_tensor_reg->getClTensorReserver(input)->descriptor);

  auto input_shape = _tensor_reg->getClTensorReserver(input)->shape;
  auto kernel_shape = _tensor_reg->getClTensorReserver(kernel)->shape;
  auto output_shape = _tensor_reg->getClTensorReserver(output)->shape;
  auto bias_shape = _tensor_reg->getClTensorReserver(bias)->shape;

  op_def.dst_tensors.push_back(_tensor_reg->getClTensorReserver(output)->descriptor);

  ModelHints hints;
  std::unique_ptr<GPUOperation> gpu_op; // = InitSingleOpSubgraph(inputs, outputs, gpu_subgraph);

  auto input_tensor = _tensor_reg->getClTensor(input);
  auto kernel_tensor = _tensor_reg->getClTensor(kernel);
  auto bias_tensor = _tensor_reg->getClTensor(bias);
  auto output_tensor = _tensor_reg->getClTensor(output);

  gpu_cl::Convolution2DAttributes attr;
  attr.strides = ToHW(param.stride.vertical, param.stride.horizontal);
  attr.dilations = HW(std::max(static_cast<u_int32_t>(1), param.dilation.height_factor),
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

  gpu_op = SelectConvolution(attr, output_shape, _creation_context->GetDeviceInfo(), op_def, hints);
  gpu_op->SetSrc(input_tensor->handle(), ir::operation::Conv2D::INPUT);

  auto fn = std::make_unique<ClFunction>();

  fn->configure(_creation_context);

  const auto activation = node.param().activation;

  switch (activation)
  {
    case ir::Activation::NONE:
    {
      gpu_op->SetDst(output_tensor->handle(), 0);
      fn->add_operation(std::move(gpu_op));
      break;
    }
    case ir::Activation::RELU6:
    {
      std::unique_ptr<GPUOperation> gpu_op_1;
      OperationDef op_def_1;
      std::shared_ptr<Tensor> new_tensor = std::make_shared<Tensor>();

      _new_tensors[output] = new_tensor;
      if (!CreateTensor(*_creation_context->context, output_shape,
                        _tensor_reg->getClTensorReserver(output)->descriptor, new_tensor.get())
             .ok())
      {
        throw std::runtime_error("Error CreateTensor.");
      }

      gpu_op->SetDst(new_tensor.get(), 0);
      fn->add_operation(std::move(gpu_op));
      op_def_1.precision = CalculationsPrecision::F32;
      op_def_1.src_tensors.push_back(_tensor_reg->getClTensorReserver(output)->descriptor);
      op_def_1.dst_tensors.push_back(_tensor_reg->getClTensorReserver(output)->descriptor);

      //   - ReLU6: clip = 6, alpha = 0
      ReLUAttributes attr_1;
      attr_1.clip = 6;
      attr_1.alpha = 0;
      gpu_op_1 = SelectReLU(attr_1, op_def_1);

      gpu_op_1->SetSrc(new_tensor.get(), 0);
      gpu_op_1->SetDst(output_tensor->handle(), 0);
      fn->add_operation(std::move(gpu_op_1));
      break;
    }
    default:
    {
      throw std::runtime_error("gpu_cl KernelGenerator : Not supported operation yet");
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
  const auto padding = node.param().padding;

  const auto multiplier = node.param().multiplier;

  auto ofm_tensor = _tensor_reg->getClTensor(ofm_index);
  auto ifm_tensor = _tensor_reg->getClTensor(ifm_index);
  auto ker_tensor = _tensor_reg->getClTensor(ker_index);
  auto bias_tensor = _tensor_reg->getClTensor(bias_index);

  bool is_weight = (_ctx.at(ker_index).isConstant() ? true : false);
  OperationDef op_def;
  op_def.precision = CalculationsPrecision::F32;

  op_def.src_tensors.push_back(_tensor_reg->getClTensorReserver(ifm_index)->descriptor);
  auto input_shape = _tensor_reg->getClTensorReserver(ifm_index)->shape;

  auto ker_shape = _tensor_reg->getClTensorReserver(ker_index)->shape;

  op_def.dst_tensors.push_back(_tensor_reg->getClTensorReserver(ofm_index)->descriptor);
  auto out_shape = _tensor_reg->getClTensorReserver(ofm_index)->shape;
  auto bias_shape = _tensor_reg->getClTensorReserver(bias_index)->shape;

  DepthwiseConvolution2DAttributes attr;
  attr.strides = ToHW(stride.vertical, stride.horizontal);
  attr.dilations = HW(std::max(static_cast<u_int32_t>(1), dilation.height_factor),
                      std::max(static_cast<u_int32_t>(1), dilation.width_factor));

  if (is_weight)
  {
    attr.weights.id = ker_index.value();
    attr.weights.shape.o = ker_shape.b;
    attr.weights.shape.h = ker_shape.h;
    attr.weights.shape.w = ker_shape.w;
    attr.weights.shape.i = ker_shape.c;
    attr.weights.data.resize(ker_shape.DimensionsProduct());
    memcpy(attr.weights.data.data(), _ctx.at(ker_index).data()->base(), ker_tensor->total_size());
  }
  attr.bias.id = bias_index.value();
  attr.bias.shape.v = bias_shape.b != 1 ? bias_shape.b : bias_shape.c;
  attr.bias.data.resize(bias_shape.DimensionsProduct());
  memcpy(attr.bias.data.data(), _ctx.at(bias_index).data()->base(), bias_tensor->total_size());
  UpdatePadding(padding.type, input_shape, &attr);

  if (multiplier != 1)
  {
    const int input_depth = input_shape.c;
    const int filter_height = ker_shape.h;
    const int filter_width = ker_shape.w;
    const int output_depth = out_shape.c;

    InternalTensor<OHWI, DataType::FLOAT32> weights;
    weights.id = attr.weights.id;
    weights.shape = OHWI(output_depth, filter_height, filter_width, input_depth);
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

  auto fn = std::make_unique<ClFunction>();
  std::unique_ptr<GPUOperation> gpu_op;

  if (is_weight)
  {
    gpu_op = SelectDWConvolution(attr, _creation_context->GetDeviceInfo(), op_def);
  }
  else
  {
    if (ker_shape.b != 1)
    {
      throw std::runtime_error(
        "No support of depthwise runtime weights with channel multiplier != 1");
    }
    gpu_op = SelectDWConvolutionDynamicWeights(attr, _creation_context->GetDeviceInfo(), op_def);
  }

  gpu_op->SetSrc(ifm_tensor->handle(), ir::operation::DepthwiseConv2D::Input::INPUT);

  fn->configure(_creation_context);

  const auto activation = node.param().activation;

  switch (activation)
  {
    case ir::Activation::NONE:
    {
      gpu_op->SetDst(ofm_tensor->handle(), 0);
      fn->add_operation(std::move(gpu_op));
      break;
    }
    case ir::Activation::RELU6:
    {
      std::unique_ptr<GPUOperation> gpu_op_1;
      OperationDef op_def_1;
      std::shared_ptr<Tensor> new_tensor = std::make_shared<Tensor>();

      _new_tensors[ofm_index] = new_tensor;
      if (!CreateTensor(*_creation_context->context, out_shape,
                        _tensor_reg->getClTensorReserver(ofm_index)->descriptor, new_tensor.get())
             .ok())
      {
        throw std::runtime_error("Error CreateTensor.");
      }

      gpu_op->SetDst(new_tensor.get(), 0);
      fn->add_operation(std::move(gpu_op));
      op_def_1.precision = CalculationsPrecision::F32;
      op_def_1.src_tensors.push_back(_tensor_reg->getClTensorReserver(ofm_index)->descriptor);
      op_def_1.dst_tensors.push_back(_tensor_reg->getClTensorReserver(ofm_index)->descriptor);

      //   - ReLU6: clip = 6, alpha = 0
      ReLUAttributes attr_1;
      attr_1.clip = 6;
      attr_1.alpha = 0;
      gpu_op_1 = SelectReLU(attr_1, op_def_1);

      gpu_op_1->SetSrc(new_tensor.get(), 0);
      gpu_op_1->SetDst(ofm_tensor->handle(), 0);
      fn->add_operation(std::move(gpu_op_1));
      break;
    }
    default:
    {
      throw std::runtime_error("gpu_cl KernelGenerator : Not supported operation yet");
    }
  }

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::ElementwiseActivation &node)
{
  std::unique_ptr<GPUOperation> gpu_op;
  auto fn = std::make_unique<ClFunction>();

  switch (node.param().op_type)
  {
    case ir::operation::ElementwiseActivation::Type::LEAKY_RELU:
    case ir::operation::ElementwiseActivation::Type::RELU:
    {
      const auto output_index{node.getOutputs().at(0)};
      const auto input_index{
        node.getInputs().at(ir::operation::ElementwiseActivation::Input::INPUT)};

      OperationDef op_def;
      op_def.precision = CalculationsPrecision::F32;
      auto output_tensor = _tensor_reg->getClTensor(output_index);
      auto input_tensor = _tensor_reg->getClTensor(input_index);
      op_def.dst_tensors.push_back(_tensor_reg->getClTensorReserver(output_index)->descriptor);
      op_def.src_tensors.push_back(_tensor_reg->getClTensorReserver(input_index)->descriptor);

      ReLUAttributes attr;
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
      gpu_op->SetSrc(input_tensor->handle(), ir::operation::ElementwiseActivation::Input::INPUT);
      gpu_op->SetDst(output_tensor->handle(), 0);
      fn->configure(_creation_context);
      fn->add_operation(std::move(gpu_op));

      _return_fn = std::move(fn);
      break;
    }
    default:
      throw std::runtime_error("gpu_cl KernelGenerator : Not supported operation yet");
  }
}

void KernelGenerator::visit(const ir::operation::Pool2D &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Pool2D::Input::INPUT)};

  OperationDef op_def;
  op_def.precision = CalculationsPrecision::F32;

  op_def.src_tensors.push_back(_tensor_reg->getClTensorReserver(input_index)->descriptor);
  auto input_shape = _tensor_reg->getClTensorReserver(input_index)->shape;

  op_def.dst_tensors.push_back(_tensor_reg->getClTensorReserver(output_index)->descriptor);

  const auto kh = node.param().kh;
  const auto kw = node.param().kw;
  const auto stride = node.param().stride;
  const auto op_type = convertPoolType(node.param().op_type);

  Pooling2DAttributes attributes;
  attributes.type = op_type;
  attributes.kernel = HW(kh > 0 ? kh : 1, kw > 0 ? kw : 1);
  attributes.strides =
    HW(stride.vertical > 0 ? stride.vertical : 1, stride.horizontal > 0 ? stride.horizontal : 1);

  if (node.param().padding.type == ir::PaddingType::SAME)
  {
    attributes.padding = CalculateSamePadding(input_shape, attributes);
  }
  else
  {
    attributes.padding.prepended = HW(0, 0);
    attributes.padding.appended = HW(0, 0);
  }

  auto fn = std::make_unique<ClFunction>();
  std::unique_ptr<GPUOperation> gpu_op;
  gpu_op = SelectPooling(attributes, op_def);

  auto input_tensor = _tensor_reg->getClTensor(input_index);
  auto output_tensor = _tensor_reg->getClTensor(output_index);

  gpu_op->SetSrc(input_tensor->handle(), ir::operation::Pool2D::Input::INPUT);
  gpu_op->SetDst(output_tensor->handle(), 0);

  fn->configure(_creation_context);
  fn->add_operation(std::move(gpu_op));

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Reshape &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Reshape::Input::INPUT)};

  OperationDef op_def;
  op_def.precision = CalculationsPrecision::F32;

  op_def.src_tensors.push_back(_tensor_reg->getClTensorReserver(input_index)->descriptor);
  auto input_shape = _tensor_reg->getClTensorReserver(input_index)->shape;

  op_def.dst_tensors.push_back(_tensor_reg->getClTensorReserver(output_index)->descriptor);
  auto output_shape = _tensor_reg->getClTensorReserver(output_index)->shape;

  ReshapeAttributes attr;
  attr.new_shape = output_shape;

  auto fn = std::make_unique<ClFunction>();
  std::unique_ptr<GPUOperation> gpu_op;
  const int src_channels = input_shape.c;
  SelectReshape(src_channels, attr.new_shape.c, op_def, &gpu_op);

  auto input_tensor = _tensor_reg->getClTensor(input_index);
  auto output_tensor = _tensor_reg->getClTensor(output_index);
  gpu_op->SetSrc(input_tensor->handle(), ir::operation::Reshape::Input::INPUT);
  gpu_op->SetDst(output_tensor->handle(), 0);

  fn->configure(_creation_context);
  fn->add_operation(std::move(gpu_op));

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

  OperationDef op_def;
  op_def.precision = CalculationsPrecision::F32;

  op_def.dst_tensors.push_back(_tensor_reg->getClTensorReserver(output_index)->descriptor);

  op_def.src_tensors.push_back(_tensor_reg->getClTensorReserver(input_index)->descriptor);
  auto input_shape = _tensor_reg->getClTensorReserver(input_index)->shape;

  auto fn = std::make_unique<ClFunction>();

  std::unique_ptr<GPUOperation> gpu_op;
  SelectSoftmax(input_shape, op_def, &gpu_op);
  auto output_tensor = _tensor_reg->getClTensor(output_index);
  auto input_tensor = _tensor_reg->getClTensor(input_index);

  gpu_op->SetSrc(input_tensor->handle(), ir::operation::Softmax::Input::INPUT);
  gpu_op->SetDst(output_tensor->handle(), 0);

  fn->configure(_creation_context);
  fn->add_operation(std::move(gpu_op));

  _return_fn = std::move(fn);
}

} // namespace gpu_cl
} // namespace backend
} // namespace onert
