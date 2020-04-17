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

#ifndef __BASE_LOADER_BASE_LOADER_H__
#define __BASE_LOADER_BASE_LOADER_H__

#include "ir/Graph.h"
#include "ir/Operations.Include.h"

#include <map>
#include <memory>
#include <fstream>
#include <limits>

namespace onert
{
namespace base_loader
{

template <typename LoaderDomain, typename SpecificLoader> class BaseLoader
{
  using Verifier = typename LoaderDomain::Verifier;
  using ActivationFunctionType = typename LoaderDomain::ActivationFunctionType;
  using Buffer = typename LoaderDomain::Buffer;
  using BuiltinOperator = typename LoaderDomain::BuiltinOperator;
  using CustomOptionsFormat = typename LoaderDomain::CustomOptionsFormat;
  using Model = typename LoaderDomain::Model;
  using Operator = typename LoaderDomain::Operator;
  using Padding = typename LoaderDomain::Padding;
  using Pool2DOptions = typename LoaderDomain::Pool2DOptions;
  using SubGraph = typename LoaderDomain::SubGraph;
  using Tensor = typename LoaderDomain::Tensor;
  using TensorType = typename LoaderDomain::TensorType;

public:
  /**
   * @brief Construct a new Loader object
   *
   * @param graph reference on primary subgraph
   */
  explicit BaseLoader(std::unique_ptr<ir::Graph> &graph) : _primary_subgraph(graph), _model{nullptr}
  {
  }

  /**
   * @brief Load a model from file
   *
   * @param file_path
   */
  void loadFromFile(const char *file_path);

protected:
  ~BaseLoader() = default;

  void loadModel();

  // Helper functions
  ir::Activation convertActivation(ActivationFunctionType type);
  ir::DataType tensorTypeToDataType(TensorType type);

  // Create operands form tflite::Tensor
  ir::OperandIndex loadOperand(const Tensor *tensor, ir::Graph &subg);
  void loadOperationIO(const Operator *op, ir::OperandIndexSequence &inputs,
                       ir::OperandIndexSequence &outputs);
  // Create operations from Operator
  void loadOperation(const Operator *op, ir::Graph &subg);
  // Load Strides and Paddings from options to param
  template <typename Param, typename OptionsType>
  void loadStridesAndPaddings(Param &param, const OptionsType *options);
  // Load Pool2D param
  template <typename Param> void loadPool2D(Param &param, const Pool2DOptions *options);

  // Operations
  void loadConv2D(const Operator *op, ir::Graph &subg);
  void loadDepthwiseConv2D(const Operator *op, ir::Graph &subg);
  void loadTransposeConv(const Operator *op, ir::Graph &subg);
  void loadAvgPool2D(const Operator *op, ir::Graph &subg);
  void loadReshape(const Operator *op, ir::Graph &subg);
  void loadSoftmax(const Operator *op, ir::Graph &subg);
  void loadMaxPool2D(const Operator *op, ir::Graph &subg);
  void loadConcatenation(const Operator *op, ir::Graph &subg);
  void loadInstanceNorm(const Operator *op, ir::Graph &subg);
  void loadFC(const Operator *op, ir::Graph &subg);
  void loadAdd(const Operator *op, ir::Graph &subg);
  void loadSub(const Operator *op, ir::Graph &subg);
  void loadMul(const Operator *op, ir::Graph &subg);
  void loadDiv(const Operator *op, ir::Graph &subg);
  void loadPack(const Operator *op, ir::Graph &subg);
  void loadRelu(const Operator *op, ir::Graph &subg);
  void loadRelu6(const Operator *op, ir::Graph &subg);
  void loadResizeBilinear(const Operator *op, ir::Graph &subg);
  void loadRsqrt(const Operator *op, ir::Graph &subg);
  void loadSqrt(const Operator *op, ir::Graph &subg);
  void loadSquaredDifference(const Operator *op, ir::Graph &subg);
  void loadTanh(const Operator *op, ir::Graph &subg);
  void loadTranspose(const Operator *op, ir::Graph &subg);
  void loadMean(const Operator *op, ir::Graph &subg);
  void loadReduceMax(const Operator *op, ir::Graph &subg);
  void loadPad(const Operator *op, ir::Graph &subg);
  void loadLogistic(const Operator *op, ir::Graph &subg);
  void loadExp(const Operator *op, ir::Graph &subg);
  void loadGather(const Operator *op, ir::Graph &subg);
  void loadCustom(const Operator *op, ir::Graph &subg);
  void loadSpaceToBatchND(const Operator *op, ir::Graph &subg);
  void loadBatchToSpaceND(const Operator *op, ir::Graph &subg);
  void loadReduceSum(const Operator *op, ir::Graph &subg);
  void loadSqueeze(const Operator *op, ir::Graph &subg);
  void loadPrelu(const Operator *op, ir::Graph &subg);
  void loadSplit(const Operator *op, ir::Graph &subg);
  void loadSlice(const Operator *op, ir::Graph &subg);
  void loadStridedSlice(const Operator *op, ir::Graph &subg);
  void loadUnpack(const Operator *op, ir::Graph &subg);
  void loadMinimum(const Operator *op, ir::Graph &subg);
  void loadMaximum(const Operator *op, ir::Graph &subg);
  void loadCast(const Operator *op, ir::Graph &subg);
  void loadComparison(const Operator *op, ir::Graph &subg);
  void loadOneHot(const Operator *op, ir::Graph &subg);
  void loadAbs(const Operator *op, ir::Graph &subg);
  void loadSin(const Operator *op, ir::Graph &subg);
  void loadShape(const Operator *op, ir::Graph &subg);

protected:
  // Buffer for loading (if needed)
  std::vector<char> _buffer;
  // Reference on loadable primary subgraph
  std::unique_ptr<ir::Graph> &_primary_subgraph;
  const Model *_model;
  // Maps Tensor indices to onert Operands.
  std::vector<ir::OperandIndex> _tensor_to_operand;
  // Verifier
  std::unique_ptr<Verifier> _verifier;
};

template <typename LoaderDomain, typename SpecificLoader>
void BaseLoader<LoaderDomain, SpecificLoader>::BaseLoader::loadFromFile(const char *file_path)
{
  std::ifstream stream(file_path, std::fstream::in | std::fstream::binary);

  if (!stream)
  {
    std::string msg = "Failed to open file `";
    msg += file_path;
    msg += "`";
    throw std::runtime_error{msg};
  }

  stream.seekg(0, stream.end);
  auto size = stream.tellg();
  stream.seekg(0, stream.beg);

  _buffer.resize(size);
  stream.read(_buffer.data(), size);

  stream.close();

  // Prepare verifier
  _verifier = std::make_unique<Verifier>(reinterpret_cast<const std::uint8_t *>(_buffer.data()),
                                         _buffer.size());

  loadModel();
}

template <typename LoaderDomain, typename SpecificLoader>
ir::Activation BaseLoader<LoaderDomain, SpecificLoader>::BaseLoader::convertActivation(
    const ActivationFunctionType type)
{
  switch (type)
  {
    case ActivationFunctionType::ActivationFunctionType_NONE:
      return ir::Activation::NONE;
    case ActivationFunctionType::ActivationFunctionType_RELU:
      return ir::Activation::RELU;
    case ActivationFunctionType::ActivationFunctionType_RELU_N1_TO_1:
      return ir::Activation::RELU1;
    case ActivationFunctionType::ActivationFunctionType_RELU6:
      return ir::Activation::RELU6;
    case ActivationFunctionType::ActivationFunctionType_TANH:
      return ir::Activation::TANH;
    default:
      throw std::runtime_error(std::string("Unsupported activation type: ")
                                   .append(EnumNameActivationFunctionType(type)));
  }
}

template <typename LoaderDomain, typename SpecificLoader>
ir::DataType
BaseLoader<LoaderDomain, SpecificLoader>::BaseLoader::tensorTypeToDataType(const TensorType type)
{
  switch (type)
  {
    case TensorType::TensorType_FLOAT32:
      return ir::DataType::FLOAT32;
    case TensorType::TensorType_INT32:
      return ir::DataType::INT32;
    case TensorType::TensorType_BOOL:
      return ir::DataType::BOOL8;
    case TensorType::TensorType_UINT8:
      return ir::DataType::QUANT8_ASYMM;
    default:
      throw std::runtime_error(
          std::string("Unsupported tensor type: ").append(EnumNameTensorType(type)));
  }
}

template <typename LoaderDomain, typename SpecificLoader>
ir::OperandIndex BaseLoader<LoaderDomain, SpecificLoader>::loadOperand(const Tensor *tensor,
                                                                       ir::Graph &subg)
{
  ir::Shape shape;
  // Shape
  const auto *tensor_shape = tensor->shape();
  if (tensor_shape != nullptr)
  {
    for (const auto &dim : *tensor_shape)
    {
      shape.append(dim);
    }
  }
  // Type
  ir::DataType data_type = tensorTypeToDataType(tensor->type());
  // Quantization
  auto q_params = tensor->quantization();
  float scale = 0.0;
  long zero_point = 0;
  if (q_params != nullptr)
  {
    if (q_params->scale())
    {
      if (q_params->scale()->size() != 1)
      {
        throw std::runtime_error("Only 1 scale for a tensor is supported.");
      }
      scale = q_params->scale()->Get(0);
    }

    if (q_params->zero_point())
    {
      if (q_params->zero_point()->size() != 1)
      {
        throw std::runtime_error("Only 1 zero_point value for a tensor is supported.");
      }
      zero_point = q_params->zero_point()->Get(0);
      // zero_point is long while TypeInfo.zero_point is defined as int32_t.
      assert(zero_point >= std::numeric_limits<int32_t>::min());
      assert(zero_point <= std::numeric_limits<int32_t>::max());
    }
    auto details = q_params->details_as_CustomQuantization();
    if (details != nullptr)
      throw std::runtime_error("Custom Quantization is not supported");
  }
  // Create TypeInfo
  ir::TypeInfo type_info(data_type, scale, zero_point);
  // Create operand
  const auto operand_index = subg.addOperand(shape, type_info);

  // Constant tensors are indicated by non-empty data.
  const auto *data = _model->buffers()->Get(tensor->buffer())->data();
  if (data != nullptr)
  {
    auto ptr = std::make_unique<ir::CachedData>(data->data(), data->size());
    subg.setOperandValue(operand_index, std::move(ptr));
  }

  // Name unused
  // auto name = tensor->name();
  // Variablie
  if (tensor->is_variable())
    throw std::runtime_error("Variable tensor not supported!");

  return operand_index;
}

template <typename LoaderDomain, typename SpecificLoader>
void BaseLoader<LoaderDomain, SpecificLoader>::loadOperationIO(const Operator *op,
                                                               ir::OperandIndexSequence &inputs,
                                                               ir::OperandIndexSequence &outputs)
{
  for (const std::int32_t idx : *op->inputs())
  {
    inputs.append(_tensor_to_operand[idx]);
  }

  for (const std::int32_t idx : *op->outputs())
  {
    outputs.append(_tensor_to_operand[idx]);
  }
}

template <typename LoaderDomain, typename SpecificLoader>
template <typename Param, typename OptionsType>
void BaseLoader<LoaderDomain, SpecificLoader>::loadStridesAndPaddings(Param &param,
                                                                      const OptionsType *options)
{
  // Strides
  param.stride.vertical = options->stride_w();
  param.stride.horizontal = options->stride_h();
  // Paddings
  if (options->padding() == Padding::Padding_SAME)
    param.padding.type = ir::PaddingType::SAME;
  if (options->padding() == Padding::Padding_VALID)
    param.padding.type = ir::PaddingType::VALID;
  // param paddings indexes unused
}

template <typename LoaderDomain, typename SpecificLoader>
template <typename Param>
void BaseLoader<LoaderDomain, SpecificLoader>::loadPool2D(Param &param,
                                                          const Pool2DOptions *options)
{
  // Strides and Paddings
  loadStridesAndPaddings(param, options);
  // Filter width and height
  // Strides
  param.kw = options->filter_width();
  param.kh = options->filter_height();
  // Activation
  param.activation = convertActivation(options->fused_activation_function());
}

template <typename LoaderDomain, typename SpecificLoader>
void BaseLoader<LoaderDomain, SpecificLoader>::loadConv2D(const Operator *op, ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);

  ir::operation::Conv2D::Param param;
  const auto *options = op->builtin_options_as_Conv2DOptions();
  param.activation = convertActivation(options->fused_activation_function());
  loadStridesAndPaddings(param, options);
  // Dilation h/w factor unused
  std::unique_ptr<ir::Operation> new_op(new ir::operation::Conv2D(inputs, outputs, param));
  subg.addOperation(std::move(new_op));
}

template <typename LoaderDomain, typename SpecificLoader>
void BaseLoader<LoaderDomain, SpecificLoader>::loadDepthwiseConv2D(const Operator *op,
                                                                   ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);

  ir::operation::DepthwiseConv2D::Param param;
  const auto *options = op->builtin_options_as_DepthwiseConv2DOptions();
  param.activation = convertActivation(options->fused_activation_function());
  loadStridesAndPaddings(param, options);
  // Multiplier
  param.multiplier = options->depth_multiplier();
  // Dilation h/w factor unused
  std::unique_ptr<ir::Operation> new_op(new ir::operation::DepthwiseConv2D(inputs, outputs, param));
  subg.addOperation(std::move(new_op));
}

template <typename LoaderDomain, typename SpecificLoader>
void BaseLoader<LoaderDomain, SpecificLoader>::loadTransposeConv(const Operator *op,
                                                                 ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);

  ir::operation::TransposeConv::Param param;
  const auto *options = op->builtin_options_as_TransposeConvOptions();
  loadStridesAndPaddings(param, options);
  std::unique_ptr<ir::Operation> new_op(new ir::operation::TransposeConv(inputs, outputs, param));
  subg.addOperation(std::move(new_op));
}

template <typename LoaderDomain, typename SpecificLoader>
void BaseLoader<LoaderDomain, SpecificLoader>::loadAvgPool2D(const Operator *op, ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);

  ir::operation::AvgPool2D::Param param;
  const auto *options = op->builtin_options_as_Pool2DOptions();

  loadPool2D(param, options);

  std::unique_ptr<ir::Operation> new_op(new ir::operation::AvgPool2D(inputs, outputs, param));
  subg.addOperation(std::move(new_op));
}

template <typename LoaderDomain, typename SpecificLoader>
void BaseLoader<LoaderDomain, SpecificLoader>::loadReshape(const Operator *op, ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);

  // const auto *options = op->builtin_options_as_ReshapeOptions();
  // No params

  std::unique_ptr<ir::Operation> new_op(new ir::operation::Reshape(inputs, outputs));
  subg.addOperation(std::move(new_op));
}

template <typename LoaderDomain, typename SpecificLoader>
void BaseLoader<LoaderDomain, SpecificLoader>::loadSoftmax(const Operator *op, ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);

  ir::operation::Softmax::Param param;
  const auto *options = op->builtin_options_as_SoftmaxOptions();
  // Beta
  param.beta = options->beta();

  std::unique_ptr<ir::Operation> new_op(new ir::operation::Softmax(inputs, outputs, param));
  subg.addOperation(std::move(new_op));
}

template <typename LoaderDomain, typename SpecificLoader>
void BaseLoader<LoaderDomain, SpecificLoader>::loadMaxPool2D(const Operator *op, ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);

  ir::operation::MaxPool2D::Param param;
  const auto *options = op->builtin_options_as_Pool2DOptions();

  loadPool2D(param, options);

  std::unique_ptr<ir::Operation> new_op(new ir::operation::MaxPool2D(inputs, outputs, param));
  subg.addOperation(std::move(new_op));
}

template <typename LoaderDomain, typename SpecificLoader>
void BaseLoader<LoaderDomain, SpecificLoader>::loadConcatenation(const Operator *op,
                                                                 ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);

  ir::operation::Concat::Param param;
  const auto *options = op->builtin_options_as_ConcatenationOptions();
  // Axis
  param.axis = options->axis();
  param.rank = subg.operands().at(outputs.at(0)).shape().rank();
  // activation unused

  std::unique_ptr<ir::Operation> new_op(new ir::operation::Concat(inputs, outputs, param));
  subg.addOperation(std::move(new_op));
}

template <typename LoaderDomain, typename SpecificLoader>
void BaseLoader<LoaderDomain, SpecificLoader>::loadInstanceNorm(const Operator *op, ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);

  ir::operation::InstanceNorm::Param param;
  const auto *options = op->builtin_options_as_InstanceNormOptions();

  param.activation = convertActivation(options->fused_activation_function());
  // Use default value 1e-5 if value of epsilon is zero
  param.epsilon = options->epsilon() == 0.f ? 1e-5 : options->epsilon();

  std::unique_ptr<ir::Operation> new_op(new ir::operation::InstanceNorm(inputs, outputs, param));
  subg.addOperation(std::move(new_op));
}

template <typename LoaderDomain, typename SpecificLoader>
void BaseLoader<LoaderDomain, SpecificLoader>::loadFC(const Operator *op, ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);

  const auto &input_operand = subg.operands().at(inputs.at(ir::operation::FullyConnected::INPUT));
  auto &weights_operand = subg.operands().at(inputs.at(ir::operation::FullyConnected::WEIGHT));
  if (input_operand.typeInfo().type() == ir::DataType::FLOAT32 &&
      weights_operand.typeInfo().type() == ir::DataType::QUANT8_ASYMM)
  {
    weights_operand.type(ir::DataType::QUANT8_SYMM);
  }

  ir::operation::FullyConnected::Param param;
  const auto *options = op->builtin_options_as_FullyConnectedOptions();

  param.activation = convertActivation(options->fused_activation_function());
  // weights_format unused

  std::unique_ptr<ir::Operation> new_op(new ir::operation::FullyConnected(inputs, outputs, param));
  subg.addOperation(std::move(new_op));
}

template <typename LoaderDomain, typename SpecificLoader>
void BaseLoader<LoaderDomain, SpecificLoader>::loadAdd(const Operator *op, ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);

  ir::operation::Add::Param param;
  const auto *options = op->builtin_options_as_AddOptions();

  param.activation = convertActivation(options->fused_activation_function());

  std::unique_ptr<ir::Operation> new_op(new ir::operation::Add(inputs, outputs, param));
  subg.addOperation(std::move(new_op));
}

template <typename LoaderDomain, typename SpecificLoader>
void BaseLoader<LoaderDomain, SpecificLoader>::loadSub(const Operator *op, ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);

  ir::operation::Sub::Param param;
  const auto *options = op->builtin_options_as_SubOptions();

  param.activation = convertActivation(options->fused_activation_function());

  std::unique_ptr<ir::Operation> new_op(new ir::operation::Sub(inputs, outputs, param));
  subg.addOperation(std::move(new_op));
}

template <typename LoaderDomain, typename SpecificLoader>
void BaseLoader<LoaderDomain, SpecificLoader>::loadMul(const Operator *op, ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);

  ir::operation::Mul::Param param;
  const auto *options = op->builtin_options_as_MulOptions();

  param.activation = convertActivation(options->fused_activation_function());

  std::unique_ptr<ir::Operation> new_op(new ir::operation::Mul(inputs, outputs, param));
  subg.addOperation(std::move(new_op));
}

template <typename LoaderDomain, typename SpecificLoader>
void BaseLoader<LoaderDomain, SpecificLoader>::loadDiv(const Operator *op, ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);

  ir::operation::Div::Param param;
  const auto *options = op->builtin_options_as_DivOptions();

  param.activation = convertActivation(options->fused_activation_function());

  std::unique_ptr<ir::Operation> new_op(new ir::operation::Div(inputs, outputs, param));
  subg.addOperation(std::move(new_op));
}

template <typename LoaderDomain, typename SpecificLoader>
void BaseLoader<LoaderDomain, SpecificLoader>::loadPack(const Operator *op, ir::Graph &subg)
{
  // This runtime_error will be removed if the one of backend supports this operation
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);

  ir::operation::Pack::Param param;
  const auto *options = op->builtin_options_as_PackOptions();
  param.num = options->values_count();
  param.axis = options->axis();
  param.rank = subg.operands().at(outputs.at(0)).shape().rank();

  std::unique_ptr<ir::Operation> new_op(new ir::operation::Pack(inputs, outputs, param));
  subg.addOperation(std::move(new_op));
}

template <typename LoaderDomain, typename SpecificLoader>
void BaseLoader<LoaderDomain, SpecificLoader>::loadRelu(const Operator *op, ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);

  std::unique_ptr<ir::Operation> new_op(new ir::operation::ReLU(inputs, outputs));
  subg.addOperation(std::move(new_op));
}

template <typename LoaderDomain, typename SpecificLoader>
void BaseLoader<LoaderDomain, SpecificLoader>::loadRelu6(const Operator *op, ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);

  std::unique_ptr<ir::Operation> new_op(new ir::operation::ReLU6(inputs, outputs));
  subg.addOperation(std::move(new_op));
}

template <typename LoaderDomain, typename SpecificLoader>
void BaseLoader<LoaderDomain, SpecificLoader>::loadResizeBilinear(const Operator *op,
                                                                  ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);
  auto input = inputs.at(0);
  auto size = inputs.at(1);

  // FIXME Handle ResizeBilinearOptions.
  if (!subg.operands().at(size).isConstant())
    throw std::runtime_error("ResizeBilinear: non-constant 'size' is not supported.");

  std::vector<std::int32_t> size_v = subg.operands().at(size).template asVector<std::int32_t>();

  ir::operation::ResizeBilinear::Param param;
  param.height_out = size_v[0];
  param.width_out = size_v[1];

  std::unique_ptr<ir::Operation> new_op(new ir::operation::ResizeBilinear({input}, outputs, param));
  subg.addOperation(std::move(new_op));
}

template <typename LoaderDomain, typename SpecificLoader>
void BaseLoader<LoaderDomain, SpecificLoader>::loadRsqrt(const Operator *op, ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);

  std::unique_ptr<ir::Operation> new_op(new ir::operation::RSQRT(inputs, outputs));
  subg.addOperation(std::move(new_op));
}

template <typename LoaderDomain, typename SpecificLoader>
void BaseLoader<LoaderDomain, SpecificLoader>::loadSqrt(const Operator *op, ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);

  std::unique_ptr<ir::Operation> new_op(new ir::operation::SQRT(inputs, outputs));
  subg.addOperation(std::move(new_op));
}

template <typename LoaderDomain, typename SpecificLoader>
void BaseLoader<LoaderDomain, SpecificLoader>::loadSquaredDifference(const Operator *op,
                                                                     ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);

  std::unique_ptr<ir::Operation> new_op(new ir::operation::SquaredDifference(inputs, outputs));
  subg.addOperation(std::move(new_op));
}

template <typename LoaderDomain, typename SpecificLoader>
void BaseLoader<LoaderDomain, SpecificLoader>::loadTanh(const Operator *op, ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);

  std::unique_ptr<ir::Operation> new_op(new ir::operation::Tanh(inputs, outputs));
  subg.addOperation(std::move(new_op));
}

template <typename LoaderDomain, typename SpecificLoader>
void BaseLoader<LoaderDomain, SpecificLoader>::loadTranspose(const Operator *op, ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);
  auto input = inputs.at(0);
  auto perm = inputs.at(1);

  if (!subg.operands().at(perm).isConstant())
    throw std::runtime_error("Transpose: non-constant 'perm' is not supported.");

  ir::operation::Transpose::Param param;
  param.perm = subg.operands().at(perm).template asVector<int>();
  param.rank = subg.operands().at(inputs.at(0)).shape().rank();

  std::unique_ptr<ir::Operation> new_op(new ir::operation::Transpose({input}, outputs, param));
  subg.addOperation(std::move(new_op));
}

template <typename LoaderDomain, typename SpecificLoader>
void BaseLoader<LoaderDomain, SpecificLoader>::loadMean(const Operator *op, ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);
  auto input = inputs.at(0);
  auto axes = inputs.at(1);

  if (!subg.operands().at(axes).isConstant())
    throw std::runtime_error("Mean: non-constant 'axes' is not supported.");

  ir::operation::Mean::Param param;
  param.axes = subg.operands().at(axes).template asVector<int>();
  param.keep_dims = op->builtin_options_as_ReducerOptions()->keep_dims();
  param.rank = subg.operands().at(inputs.at(0)).shape().rank();

  std::unique_ptr<ir::Operation> new_op(new ir::operation::Mean({input}, outputs, param));
  subg.addOperation(std::move(new_op));
}

template <typename LoaderDomain, typename SpecificLoader>
void BaseLoader<LoaderDomain, SpecificLoader>::loadReduceMax(const Operator *op, ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);
  auto input = inputs.at(0);
  auto axes = inputs.at(1);

  // FIXME Handle ReducerOptions.
  if (!subg.operands().at(axes).isConstant())
    throw std::runtime_error("ReduceSum: non-constant 'axes' is not supported.");

  ir::operation::ReduceMax::Param param;
  param.axes = subg.operands().at(axes).template asVector<int>();
  param.keep_dims = op->builtin_options_as_ReducerOptions()->keep_dims();
  param.rank = subg.operands().at(inputs.at(0)).shape().rank();

  std::unique_ptr<ir::Operation> new_op(new ir::operation::ReduceMax({input}, outputs, param));
  subg.addOperation(std::move(new_op));
}

template <typename LoaderDomain, typename SpecificLoader>
void BaseLoader<LoaderDomain, SpecificLoader>::loadPad(const Operator *op, ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);

  ir::operation::Pad::Param param;
  param.rank = subg.operands().at(inputs.at(0)).shape().rank();

  std::unique_ptr<ir::Operation> new_op(new ir::operation::Pad(inputs, outputs, param));
  subg.addOperation(std::move(new_op));
}

template <typename LoaderDomain, typename SpecificLoader>
void BaseLoader<LoaderDomain, SpecificLoader>::loadLogistic(const Operator *op, ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);

  std::unique_ptr<ir::Operation> new_op(new ir::operation::Logistic(inputs, outputs));
  subg.addOperation(std::move(new_op));
}

template <typename LoaderDomain, typename SpecificLoader>
void BaseLoader<LoaderDomain, SpecificLoader>::loadExp(const Operator *op, ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);

  std::unique_ptr<ir::Operation> new_op(new ir::operation::Exp(inputs, outputs));
  subg.addOperation(std::move(new_op));
}

template <typename LoaderDomain, typename SpecificLoader>
void BaseLoader<LoaderDomain, SpecificLoader>::loadGather(const Operator *op, ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);
  ir::operation::Gather::Param param;
  param.axis = op->builtin_options_as_GatherOptions()->axis();
  param.rank = subg.operands().at(inputs.at(0)).shape().rank();

  std::unique_ptr<ir::Operation> new_op(new ir::operation::Gather(inputs, outputs, param));
  subg.addOperation(std::move(new_op));
}

template <typename LoaderDomain, typename SpecificLoader>
void BaseLoader<LoaderDomain, SpecificLoader>::loadSpaceToBatchND(const Operator *op,
                                                                  ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);

  std::unique_ptr<ir::Operation> new_op{new ir::operation::SpaceToBatchND{inputs, outputs}};
  subg.addOperation(std::move(new_op));
}

template <typename LoaderDomain, typename SpecificLoader>
void BaseLoader<LoaderDomain, SpecificLoader>::loadBatchToSpaceND(const Operator *op,
                                                                  ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);
  auto input = inputs.at(0);
  auto block_shape = inputs.at(1);
  auto crops = inputs.at(2);

  if (!subg.operands().at(crops).isConstant())
    throw std::runtime_error("BatchToSpaceND: non-constant 'crops' is not supported.");

  std::vector<std::int32_t> crops_v = subg.operands().at(crops).template asVector<std::int32_t>();
  assert(crops_v.size() == 4);
  if (crops_v != std::vector<std::int32_t>{0, 0, 0, 0})
    throw std::runtime_error("BatchToSpaceND: 'crops' other than {0, 0, 0, 0} is not supported.");

  std::unique_ptr<ir::Operation> new_op{
      new ir::operation::BatchToSpaceND{{input, block_shape}, outputs}};
  subg.addOperation(std::move(new_op));
}

template <typename LoaderDomain, typename SpecificLoader>
void BaseLoader<LoaderDomain, SpecificLoader>::loadReduceSum(const Operator *op, ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);
  auto input = inputs.at(0);
  auto axes = inputs.at(1);

  // FIXME Handle ReducerOptions.
  if (!subg.operands().at(axes).isConstant())
    throw std::runtime_error("ReduceSum: non-constant 'axes' is not supported.");

  ir::operation::ReduceSum::Param param;
  param.axes = subg.operands().at(axes).template asVector<int>();
  param.keep_dims = op->builtin_options_as_ReducerOptions()->keep_dims();
  param.rank = subg.operands().at(inputs.at(0)).shape().rank();

  std::unique_ptr<ir::Operation> new_op{new ir::operation::ReduceSum{{input}, outputs, param}};
  subg.addOperation(std::move(new_op));
}

template <typename LoaderDomain, typename SpecificLoader>
void BaseLoader<LoaderDomain, SpecificLoader>::loadCustom(const Operator *op, ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);

  auto *op_code = _model->operator_codes()->Get(op->opcode_index());
  auto custom_op_id = op_code->custom_code()->str();

  auto constraint = ir::OperandConstraint::createExact(inputs.size());

  assert(op->custom_options_format() == CustomOptionsFormat::CustomOptionsFormat_FLEXBUFFERS &&
         "Unsupported custom operation options format");

  size_t custom_op_data_size = op->custom_options()->size();
  auto custom_op_data = new char[custom_op_data_size];
  std::copy(op->custom_options()->begin(), op->custom_options()->end(), custom_op_data);

  ir::operation::Custom::Userdata userdata{};
  userdata.data = custom_op_data;
  userdata.size = custom_op_data_size;

  auto new_op =
      std::make_unique<ir::operation::Custom>(constraint, inputs, outputs, custom_op_id, userdata);

  subg.addOperation(std::move(new_op));
}

template <typename LoaderDomain, typename SpecificLoader>
void BaseLoader<LoaderDomain, SpecificLoader>::loadSqueeze(const Operator *op, ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);

  ir::operation::Squeeze::Param param{};
  const auto *options = op->builtin_options_as_SqueezeOptions();
  const auto *dims = options->squeeze_dims();
  if (dims)
  {
    if (dims->Length() > sizeof(param.dims) / sizeof(param.dims[0]))
      throw std::runtime_error("Squeeze: 'param.ndims' is out of range.");
    param.ndim = dims->Length();
    for (int i = 0; i < param.ndim; ++i)
      param.dims[i] = dims->Get(i);
  }

  std::unique_ptr<ir::Operation> new_op(new ir::operation::Squeeze(inputs, outputs, param));
  subg.addOperation(std::move(new_op));
}

template <typename LoaderDomain, typename SpecificLoader>
void BaseLoader<LoaderDomain, SpecificLoader>::loadPrelu(const Operator *op, ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);

  std::unique_ptr<ir::Operation> new_op(new ir::operation::PReLU(inputs, outputs));
  subg.addOperation(std::move(new_op));
}

template <typename LoaderDomain, typename SpecificLoader>
void BaseLoader<LoaderDomain, SpecificLoader>::loadSplit(const Operator *op, ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);
  // Notice : input order is strange for tflite split
  auto input = inputs.at(1);
  auto axis = inputs.at(0);

  // FIXME Handle SplitOptions.
  if (!subg.operands().at(axis).isConstant())
    throw std::runtime_error("Split: non-constant 'axis' is not supported.");

  ir::operation::Split::Param param{};
  param.axis = subg.operands().at(axis).template asScalar<int>();
  const auto *options = op->builtin_options_as_SplitOptions();
  param.num_splits = options->num_splits();
  param.rank = subg.operands().at(input).shape().rank();

  std::unique_ptr<ir::Operation> new_op(new ir::operation::Split({input}, outputs, param));
  subg.addOperation(std::move(new_op));
}

template <typename LoaderDomain, typename SpecificLoader>
void BaseLoader<LoaderDomain, SpecificLoader>::loadSlice(const Operator *op, ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);

  ir::operation::Slice::Param param;
  param.rank = subg.operands().at(inputs.at(0)).shape().rank();

  std::unique_ptr<ir::Operation> new_op{new ir::operation::Slice{inputs, outputs, param}};
  subg.addOperation(std::move(new_op));
}

template <typename LoaderDomain, typename SpecificLoader>
void BaseLoader<LoaderDomain, SpecificLoader>::loadStridedSlice(const Operator *op, ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);

  ir::operation::StridedSlice::Param param;

  const auto *options = op->builtin_options_as_StridedSliceOptions();
  param.begin_mask = options->begin_mask();
  param.end_mask = options->end_mask();
  param.shrink_axis_mask = options->shrink_axis_mask();
  param.rank = subg.operands().at(inputs.at(0)).shape().rank();

  std::unique_ptr<ir::Operation> new_op{new ir::operation::StridedSlice{inputs, outputs, param}};
  subg.addOperation(std::move(new_op));
}

template <typename LoaderDomain, typename SpecificLoader>
void BaseLoader<LoaderDomain, SpecificLoader>::loadUnpack(const Operator *op, ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);

  ir::operation::Unpack::Param param;
  const auto *options = op->builtin_options_as_UnpackOptions();
  param.num = options->num();
  param.axis = options->axis();
  param.rank = subg.operands().at(inputs.at(0)).shape().rank();

  std::unique_ptr<ir::Operation> new_op(new ir::operation::Unpack(inputs, outputs, param));
  subg.addOperation(std::move(new_op));
}

template <typename LoaderDomain, typename SpecificLoader>
void BaseLoader<LoaderDomain, SpecificLoader>::loadMinimum(const Operator *op, ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);

  std::unique_ptr<ir::Operation> new_op(new ir::operation::Min(inputs, outputs));
  subg.addOperation(std::move(new_op));
}

template <typename LoaderDomain, typename SpecificLoader>
void BaseLoader<LoaderDomain, SpecificLoader>::loadMaximum(const Operator *op, ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);

  std::unique_ptr<ir::Operation> new_op(new ir::operation::Max(inputs, outputs));
  subg.addOperation(std::move(new_op));
}

template <typename LoaderDomain, typename SpecificLoader>
void BaseLoader<LoaderDomain, SpecificLoader>::loadCast(const Operator *op, ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);

  auto qasymm8ToUint8 = [](ir::Operand &operand) {
    if (operand.typeInfo().type() == ir::DataType::QUANT8_ASYMM)
    {
      operand.type(ir::DataType::UINT8);
    }
  };
  qasymm8ToUint8(subg.operands().at(inputs.at(ir::operation::Cast::Input::INPUT)));
  qasymm8ToUint8(subg.operands().at(outputs.at(0)));

  std::unique_ptr<ir::Operation> new_op(new ir::operation::Cast(inputs, outputs));
  subg.addOperation(std::move(new_op));
}

template <typename LoaderDomain, typename SpecificLoader>
void BaseLoader<LoaderDomain, SpecificLoader>::loadComparison(const Operator *op, ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);

  ir::operation::Comparison::Param param;

  const auto builtin_op = _model->operator_codes()->Get(op->opcode_index())->builtin_code();

  switch (builtin_op)
  {
    case BuiltinOperator::BuiltinOperator_EQUAL:
      param.comparison_type = ir::operation::Comparison::ComparisonType::Equal;
      break;
    case BuiltinOperator::BuiltinOperator_NOT_EQUAL:
      param.comparison_type = ir::operation::Comparison::ComparisonType::NotEqual;
      break;
    case BuiltinOperator::BuiltinOperator_GREATER_EQUAL:
      param.comparison_type = ir::operation::Comparison::ComparisonType::GreaterEqual;
      break;
    case BuiltinOperator::BuiltinOperator_GREATER:
      param.comparison_type = ir::operation::Comparison::ComparisonType::Greater;
      break;
    case BuiltinOperator::BuiltinOperator_LESS_EQUAL:
      param.comparison_type = ir::operation::Comparison::ComparisonType::LessEqual;
      break;
    case BuiltinOperator::BuiltinOperator_LESS:
      param.comparison_type = ir::operation::Comparison::ComparisonType::Less;
      break;
    default:
      throw std::runtime_error(
          std::string("Unsupported operation: ").append(EnumNameBuiltinOperator(builtin_op)));
  }

  std::unique_ptr<ir::Operation> new_op(new ir::operation::Comparison(inputs, outputs, param));
  subg.addOperation(std::move(new_op));
}

template <typename LoaderDomain, typename SpecificLoader>
void BaseLoader<LoaderDomain, SpecificLoader>::loadOneHot(const Operator *op, ir::Graph &subg)
{
  if (op->inputs()->size() != 4 || op->outputs()->size() != 1)
    throw std::runtime_error("OneHot Op has wrong number of input or output tensors.");

  enum
  {
    INDICES = 0,
    DEPTH = 1,
    ON_VALUE = 2,
    OFF_VALUE = 3,
  };

  // Set input and output tensors
  ir::OperandIndexSequence inputs, outputs;
  inputs.append(_tensor_to_operand[op->inputs()->Get(INDICES)]);
  outputs.append(_tensor_to_operand[op->outputs()->Get(0)]);

  // Set parameters
  // depth, on_value and off_value are scalar though it is passed as inputs
  auto depth_opidx = _tensor_to_operand[op->inputs()->Get(DEPTH)];
  auto on_value_opidx = _tensor_to_operand[op->inputs()->Get(ON_VALUE)];
  auto off_value_opidx = _tensor_to_operand[op->inputs()->Get(OFF_VALUE)];
  const auto depth = subg.operands().at(depth_opidx).template asScalar<int>();
  const auto on_value = subg.operands().at(on_value_opidx).template asScalar<float>();
  const auto off_value = subg.operands().at(off_value_opidx).template asScalar<float>();
  const auto axis = op->builtin_options_as_OneHotOptions()->axis();
  std::unique_ptr<ir::Operation> new_op(
      new ir::operation::OneHot(inputs, outputs, {depth, on_value, off_value, axis}));
  subg.addOperation(std::move(new_op));
}

template <typename LoaderDomain, typename SpecificLoader>
void BaseLoader<LoaderDomain, SpecificLoader>::loadAbs(const Operator *op, ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);

  std::unique_ptr<ir::Operation> new_op(new ir::operation::Abs(inputs, outputs));
  subg.addOperation(std::move(new_op));
}

template <typename LoaderDomain, typename SpecificLoader>
void BaseLoader<LoaderDomain, SpecificLoader>::loadSin(const Operator *op, ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);

  std::unique_ptr<ir::Operation> new_op(new ir::operation::Sin(inputs, outputs));
  subg.addOperation(std::move(new_op));
}

template <typename LoaderDomain, typename SpecificLoader>
void BaseLoader<LoaderDomain, SpecificLoader>::loadShape(const Operator *op, ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);

  // ir::operation::Shape::Param param;
  // const auto *options = op->builtin_options_as_ShapeOptions();
  // param.out_type = tensorTypeToDataType(options->out_type());

  std::unique_ptr<ir::Operation> new_op(new ir::operation::Shape(inputs, outputs /*, param*/));
  subg.addOperation(std::move(new_op));
}

template <typename LoaderDomain, typename SpecificLoader>
void BaseLoader<LoaderDomain, SpecificLoader>::loadOperation(const Operator *op, ir::Graph &subg)
{
  const auto builtin_op = _model->operator_codes()->Get(op->opcode_index())->builtin_code();

  switch (builtin_op)
  {
    case BuiltinOperator::BuiltinOperator_CONV_2D:
      loadConv2D(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_AVERAGE_POOL_2D:
      loadAvgPool2D(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_DEPTHWISE_CONV_2D:
      loadDepthwiseConv2D(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_TRANSPOSE_CONV:
      loadTransposeConv(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_RESHAPE:
      loadReshape(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_SOFTMAX:
      loadSoftmax(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_MAX_POOL_2D:
      loadMaxPool2D(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_CONCATENATION:
      loadConcatenation(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_FULLY_CONNECTED:
      loadFC(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_ADD:
      loadAdd(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_SUB:
      loadSub(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_MUL:
      loadMul(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_DIV:
      loadDiv(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_PACK:
      loadPack(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_RELU:
      loadRelu(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_RELU6:
      loadRelu6(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_RESIZE_BILINEAR:
      loadResizeBilinear(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_RSQRT:
      loadRsqrt(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_SQRT:
      loadSqrt(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_SQUARED_DIFFERENCE:
      loadSquaredDifference(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_TANH:
      loadTanh(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_TRANSPOSE:
      loadTranspose(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_MEAN:
      loadMean(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_REDUCE_MAX:
      loadReduceMax(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_PAD:
      loadPad(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_LOGISTIC:
      loadLogistic(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_EXP:
      loadExp(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_GATHER:
      loadGather(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_SPACE_TO_BATCH_ND:
      loadSpaceToBatchND(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_BATCH_TO_SPACE_ND:
      loadBatchToSpaceND(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_SUM:
      loadReduceSum(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_CUSTOM:
      loadCustom(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_SQUEEZE:
      loadSqueeze(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_PRELU:
      loadPrelu(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_SPLIT:
      loadSplit(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_SLICE:
      loadSlice(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_STRIDED_SLICE:
      loadStridedSlice(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_UNPACK:
      loadUnpack(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_MINIMUM:
      loadMinimum(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_MAXIMUM:
      loadMaximum(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_CAST:
      loadCast(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_EQUAL:
    case BuiltinOperator::BuiltinOperator_NOT_EQUAL:
    case BuiltinOperator::BuiltinOperator_GREATER_EQUAL:
    case BuiltinOperator::BuiltinOperator_GREATER:
    case BuiltinOperator::BuiltinOperator_LESS_EQUAL:
    case BuiltinOperator::BuiltinOperator_LESS:
      loadComparison(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_ONE_HOT:
      loadOneHot(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_ABS:
      loadAbs(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_SIN:
      loadSin(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_SHAPE:
      loadShape(op, subg);
      return;
    // TODO Implement loading subgraphs of conftrol flow ops
    default:
      throw std::runtime_error(
          std::string("Unsupported operation: ").append(EnumNameBuiltinOperator(builtin_op)));
  }
}

template <typename LoaderDomain, typename SpecificLoader>
void BaseLoader<LoaderDomain, SpecificLoader>::loadModel()
{
  LoaderDomain::VerifyModelBuffer(*_verifier.get());
  _model = LoaderDomain::GetModel(_buffer.data());
  // Version unused
  // const auto version = _model->version();
  // Description unused
  // const auto *description = _model->description();
  // Metabuffer unsued
  // const auto *metadata_buffer = _model->metadata_buffer();
  // Load subgraphs recursively from primary subgraph and map operations on subgraph
  const auto domain_subgraph = (*_model->subgraphs())[0];
  _primary_subgraph = static_cast<SpecificLoader *>(this)->loadSubgraph(domain_subgraph);
}

} // namespace base_loader
} // namespace onert

#endif //__BASE_LOADER_BASE_LOADER_H__
