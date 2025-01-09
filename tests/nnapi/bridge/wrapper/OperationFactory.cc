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

#include "OperationFactory.h"
#include "NNAPIConvert.h"

#include <ir/Operations.Include.h>
#include <string.h>

namespace
{
using namespace onert::ir;

void replaceDataType(Operands &operands, const OperandIndex &index, const DataType type)
{
  assert(operands.exist(index));
  operands.at(index).type(type);
}

ExplicitPadding makeExplicitPadding(Operands &operands, const OperandIndex &left_index,
                                    const OperandIndex &right_index, const OperandIndex &top_index,
                                    const OperandIndex &bottom_index)
{
  auto left = operands.at(left_index).asScalar<int32_t>();
  auto right = operands.at(right_index).asScalar<int32_t>();
  auto top = operands.at(top_index).asScalar<int32_t>();
  auto bottom = operands.at(bottom_index).asScalar<int32_t>();

  if (left < 0 || right < 0 || top < 0 || bottom < 0)
  {
    throw std::runtime_error{"Cannot handle negative explicit padding value"};
  }

  ExplicitPadding param;
  param.left = static_cast<uint32_t>(left);
  param.right = static_cast<uint32_t>(right);
  param.top = static_cast<uint32_t>(top);
  param.bottom = static_cast<uint32_t>(bottom);

  return param;
}

Stride makeStride(Operands &operands, const OperandIndex &horizontal_index,
                  const OperandIndex &vertical_index)
{
  auto horizontal = operands.at(horizontal_index).asScalar<int32_t>();
  auto vertical = operands.at(vertical_index).asScalar<int32_t>();

  if (vertical < 0 || horizontal < 0)
  {
    throw std::runtime_error{"Cannot handle negative stride value"};
  }

  Stride stride;
  stride.horizontal = static_cast<uint32_t>(horizontal);
  stride.vertical = static_cast<uint32_t>(vertical);

  return stride;
}

uint32_t getUint32Scalar(Operands &operands, const OperandIndex index)
{
  auto int32_value = operands.at(index).asScalar<int32_t>();
  if (int32_value < 0)
  {
    throw std::runtime_error{"Cannot handle negative value"};
  }

  return static_cast<uint32_t>(int32_value);
}

Activation getActivation(Operands &operands, const OperandIndex index)
{
  switch (operands.at(index).asScalar<int32_t>())
  {
    case 0:
      return Activation::NONE;
    case 1:
      return Activation::RELU;
    case 2:
      return Activation::RELU1;
    case 3:
      return Activation::RELU6;
    case 4:
      return Activation::TANH;
    case 6:
      return Activation::SIGMOID;
    default:
      throw std::runtime_error("Unsupported activation type");
  }
}

OperationFactory::Generator
getElementwiseActivationGenerator(const onert::ir::operation::ElementwiseActivation::Type op_type,
                                  float alpha = 0.f, float beta = 0.f)
{
  return [op_type, alpha, beta](const OperationFactory::Param &init_param, Operands &) {
    assert(init_param.input_count == 1);
    assert(init_param.output_count == 1);

    // Each input should be interpreted as follows:
    //
    //  0 -> Input Tensor Index

    OperandIndexSequence inputs{init_param.inputs[0]};
    OperandIndexSequence outputs{init_param.outputs[0]};

    operation::ElementwiseActivation::Param param;
    param.op_type = op_type;
    param.alpha = alpha;
    param.beta = beta;

    return new operation::ElementwiseActivation{inputs, outputs, param};
  };
}

OperationFactory::Generator getElementwiseBinaryGenerator(
  const onert::ir::operation::ElementwiseBinary::ElementwiseBinaryType op_type)
{
  return [op_type](const OperationFactory::Param &init_param, Operands &) {
    assert(init_param.input_count == 2);
    assert(init_param.output_count == 1);

    // Each input should be interpreted as follows:
    //
    //  0 -> Lefthand side operand
    //  1 -> Righthand side operand

    OperandIndexSequence inputs{init_param.inputs[0], init_param.inputs[1]};
    OperandIndexSequence outputs{init_param.outputs[0]};

    operation::ElementwiseBinary::Param param;
    param.op_type = op_type;

    return new operation::ElementwiseBinary{inputs, outputs, param};
  };
}

OperationFactory::Generator
getElementwiseUnaryGenerator(const onert::ir::operation::ElementwiseUnary::Type op_type)
{
  return [op_type](const OperationFactory::Param &init_param, Operands &operands) {
    assert(init_param.input_count == 1);
    assert(init_param.output_count == 1);

    // Each input should be interpreted as follows:
    //
    //  0 ->  Input Tensor Index

    OperandIndexSequence inputs{init_param.inputs[0]};
    OperandIndexSequence outputs{init_param.outputs[0]};

    operation::ElementwiseUnary::Param param;
    param.op_type = op_type;

    if (op_type == operation::ElementwiseUnary::Type::CAST)
    {
      // NNAPI uses QUANT_UINT8_ASYMM to represent UINT8 type for ANEURALNETWORKS_CAST's
      // input/output
      if (operands.at(inputs.at(0)).typeInfo().type() == DataType::QUANT_UINT8_ASYMM)
      {
        replaceDataType(operands, inputs.at(0), DataType::UINT8);
      }
      if (operands.at(outputs.at(0)).typeInfo().type() == DataType::QUANT_UINT8_ASYMM)
      {
        replaceDataType(operands, outputs.at(0), DataType::UINT8);
      }
    }

    return new operation::ElementwiseUnary{inputs, outputs, param};
  };
}

OperationFactory::Generator
getBinaryArithmeticGenerator(const onert::ir::operation::BinaryArithmetic::ArithmeticType op_type)
{
  return [op_type](const OperationFactory::Param &init_param, Operands &operands) {
    assert(init_param.input_count == 3);
    assert(init_param.output_count == 1);

    // Each input should be interpreted as follows:
    //
    //  0 -> Lefthand side operand
    //  1 -> Righthand side operand

    OperandIndexSequence inputs{init_param.inputs[0], init_param.inputs[1]};
    OperandIndexSequence outputs{init_param.outputs[0]};

    operation::BinaryArithmetic::Param param;
    param.arithmetic_type = op_type;
    const auto activation_index = OperandIndex{init_param.inputs[2]};
    param.activation =
      NNAPIConvert::getFusedActivation(operands.at(activation_index).asScalar<FuseCode>());

    return new operation::BinaryArithmetic{inputs, outputs, param};
  };
}

OperationFactory::Generator
getPool2DGenerator(const onert::ir::operation::Pool2D::PoolType pool_type)
{
  return [pool_type](const OperationFactory::Param &init_param, Operands &operands) {
    assert(init_param.input_count == 7 || init_param.input_count == 10);
    assert(init_param.output_count == 1);

    // In common
    //  0 -> IFM Tensor Index
    OperandIndexSequence inputs{init_param.inputs[0]};
    OperandIndexSequence outputs{init_param.outputs[0]};

    operation::Pool2D::Param param;
    param.op_type = pool_type;
    if (init_param.input_count == 7) // support implicit padding
    {
      // Each input should be interpreted as follows:
      //
      //  1 -> Padding Code (ANEURALNETWORKS_PADDING_SAME or ANEURALNETWORKS_PADDING_VALID) Index
      //  2 -> Horizontal (over width) Stride Index
      //  3 -> Vertial (over height) Stride Index
      //  4 -> Filter Width Index
      //  5 -> Filter Height Index
      //  6 -> FuseCode (activation) Index

      const auto padding_index = OperandIndex{init_param.inputs[1]};
      const auto hstride_index = OperandIndex{init_param.inputs[2]};
      const auto vstride_index = OperandIndex{init_param.inputs[3]};
      const auto kw_index = OperandIndex{init_param.inputs[4]};
      const auto kh_index = OperandIndex{init_param.inputs[5]};
      const auto activation_index = OperandIndex{init_param.inputs[6]};

      param.padding.type =
        NNAPIConvert::getPaddingType(operands.at(padding_index).asScalar<PaddingCode>());
      param.stride = makeStride(operands, hstride_index, vstride_index);
      param.kw = getUint32Scalar(operands, kw_index);
      param.kh = operands.at(kh_index).asScalar<uint32_t>();
      param.activation =
        NNAPIConvert::getFusedActivation(operands.at(activation_index).asScalar<FuseCode>());
    }
    else // support explicit padding
    {
      // Each input should be interpreted as follows:
      //
      //  1 -> Padding_left index
      //  2 -> Padding_right index
      //  3 -> Padding_top index
      //  4 -> Padding_bottom index
      //  5 -> Horizontal (over width) Stride Index
      //  6 -> Vertial (over height) Stride Index
      //  7 -> Filter Width Index
      //  8 -> Filter Height Index
      //  9 -> FuseCode (activation) Index

      const auto padding_left_index = OperandIndex{init_param.inputs[1]};
      const auto padding_right_index = OperandIndex{init_param.inputs[2]};
      const auto padding_top_index = OperandIndex{init_param.inputs[3]};
      const auto padding_bottom_index = OperandIndex{init_param.inputs[4]};
      const auto hstride_index = OperandIndex{init_param.inputs[5]};
      const auto vstride_index = OperandIndex{init_param.inputs[6]};
      const auto kw_index = OperandIndex{init_param.inputs[7]};
      const auto kh_index = OperandIndex{init_param.inputs[8]};
      const auto activation_index = OperandIndex{init_param.inputs[9]};

      param.padding.type = PaddingType::EXPLICIT;
      param.padding.param = makeExplicitPadding(operands, padding_left_index, padding_right_index,
                                                padding_top_index, padding_bottom_index);
      param.stride = makeStride(operands, hstride_index, vstride_index);
      param.kw = getUint32Scalar(operands, kw_index);
      param.kh = getUint32Scalar(operands, kh_index);
      param.activation =
        NNAPIConvert::getFusedActivation(operands.at(activation_index).asScalar<FuseCode>());
    }
    return new operation::Pool2D{inputs, outputs, param};
  };
}

OperationFactory::Generator
getReduceGenerator(const onert::ir::operation::Reduce::ReduceType reduce_type)
{
  return [reduce_type](const OperationFactory::Param &init_param, Operands &operands) {
    assert(init_param.input_count == 3);
    assert(init_param.output_count == 1);

    // Each input should be interpreted as follows:
    //
    //  0 -> Input Tensor Index
    //  1 -> Reduced Axes Tensor Index
    //  2 -> keep_dims Index

    OperandIndexSequence inputs{init_param.inputs[0], init_param.inputs[1]};
    OperandIndexSequence outputs{init_param.outputs[0]};

    operation::Reduce::Param param;
    param.reduce_type = reduce_type;
    param.keep_dims = operands.at(OperandIndex{init_param.inputs[2]}).asScalar<int8_t>() != 0;

    return new operation::Reduce{inputs, outputs, param};
  };
}

template <typename T>
Operation *CreateSimpleUnaryOp(const OperationFactory::Param &init_param, Operands &)
{
  assert(init_param.input_count == 1 && init_param.output_count == 1);

  OperandIndexSequence outputs{init_param.outputs[0]};

  // Each input should be interpreted as follows:
  //
  //  0 -> Input Tensor Index
  OperandIndexSequence inputs{init_param.inputs[0]};

  return new T{inputs, outputs};
}

// A generator function for binary ops with no params
template <typename T>
Operation *createSimpleBinaryOp(const OperationFactory::Param &init_param, Operands &)
{
  assert(init_param.input_count == 2 && init_param.output_count == 1);

  OperandIndexSequence inputs{init_param.inputs[0], init_param.inputs[1]};
  OperandIndexSequence outputs{init_param.outputs[0]};

  return new T{inputs, outputs};
}

OperationFactory::Generator getComparisonGenerator(operation::Comparison::ComparisonType type)
{
  return [type](const OperationFactory::Param &init_param, Operands &) -> Operation * {
    assert(init_param.input_count == 2 && init_param.output_count == 1);

    OperandIndexSequence outputs{init_param.outputs[0]};

    // Each input should be interpreted as follows:
    //
    //  0 -> input0 Tensor Index
    //  1 -> input1 Tensor Index
    OperandIndexSequence inputs{init_param.inputs[0], init_param.inputs[1]};

    operation::Comparison::Param param;
    param.comparison_type = type;

    return new operation::Comparison{inputs, outputs, param};
  };
}

} // namespace

OperationFactory &OperationFactory::get()
{
  static OperationFactory factory;
  return factory;
}

OperationFactory::OperationFactory()
{
  // Each input should be interpreted as follows:
  //  0 -> Input Tensor Index
  //  1 -> Block size Index
  _map[ANEURALNETWORKS_BATCH_TO_SPACE_ND] = createSimpleBinaryOp<operation::BatchToSpaceND>;

  _map[ANEURALNETWORKS_DEPTHWISE_CONV_2D] = [](const OperationFactory::Param &init_param,
                                               Operands &operands) {
    assert((init_param.input_count == 8 || init_param.input_count == 11) &&
           init_param.output_count == 1);

    // In common
    // 0 -> IFM Tensor Index
    // 1 -> Kernel Tensor Index
    // 2 -> Bias Tensor Index
    OperandIndexSequence inputs{init_param.inputs[0], init_param.inputs[1], init_param.inputs[2]};
    OperandIndexSequence outputs{init_param.outputs[0]};

    operation::DepthwiseConv2D::Param param;
    if (init_param.input_count == 8)
    {
      // Imlicit Padding case
      // Each input should be interpreted as follows:
      //
      // 3 -> Padding Code (ANEURALNETWORKS_PADDING_SAME or ANEURALNETWORKS_PADDING_VALID) Index
      // 4 -> Stride (width) Index
      // 5 -> Stride (height) INdex
      // 6 -> Depthwise multiplier
      // 7 -> Activation Index

      const auto padding_index = OperandIndex{init_param.inputs[3]};
      const auto hstride_index = OperandIndex{init_param.inputs[4]};
      const auto vstride_index = OperandIndex{init_param.inputs[5]};
      const auto multiplier_index = OperandIndex{init_param.inputs[6]};
      const auto activation_index = OperandIndex{init_param.inputs[7]};

      param.padding.type =
        NNAPIConvert::getPaddingType(operands.at(padding_index).asScalar<PaddingCode>());
      param.stride = makeStride(operands, hstride_index, vstride_index);
      param.multiplier = getUint32Scalar(operands, multiplier_index);
      param.activation =
        NNAPIConvert::getFusedActivation(operands.at(activation_index).asScalar<FuseCode>());
    }
    else
    {
      // Explicit Padding case
      // Each input should be interpreted as follows:
      //
      // 3 -> Padding On the Left
      // 4 -> Padding On the Right
      // 5 -> Padding On the Top
      // 6 -> Padding On the Bottom
      // 7 -> Stride (width) Index
      // 8 -> Stride (height) Index
      // 9 -> Depthwise multiplier
      // 10-> Activation Index

      const auto padding_left_index = OperandIndex{init_param.inputs[3]};
      const auto padding_right_index = OperandIndex{init_param.inputs[4]};
      const auto padding_top_index = OperandIndex{init_param.inputs[5]};
      const auto padding_bottom_index = OperandIndex{init_param.inputs[6]};
      const auto hstride_index = OperandIndex{init_param.inputs[7]};
      const auto vstride_index = OperandIndex{init_param.inputs[8]};
      const auto multiplier_index = OperandIndex{init_param.inputs[9]};
      const auto activation_index = OperandIndex{init_param.inputs[10]};

      param.padding.type = PaddingType::EXPLICIT;
      param.padding.param = makeExplicitPadding(operands, padding_left_index, padding_right_index,
                                                padding_top_index, padding_bottom_index);
      param.stride = makeStride(operands, hstride_index, vstride_index);
      param.multiplier = getUint32Scalar(operands, multiplier_index);
      param.activation =
        NNAPIConvert::getFusedActivation(operands.at(activation_index).asScalar<FuseCode>());
    }

    // TODO set dilation
    param.dilation.width_factor = 1;
    param.dilation.height_factor = 1;

    return new operation::DepthwiseConv2D{inputs, outputs, param};
  };

  _map[ANEURALNETWORKS_MAX_POOL_2D] = getPool2DGenerator(operation::Pool2D::PoolType::MAX);

  _map[ANEURALNETWORKS_AVERAGE_POOL_2D] = getPool2DGenerator(operation::Pool2D::PoolType::AVG);

  _map[ANEURALNETWORKS_CONCATENATION] = [](const OperationFactory::Param &init_param,
                                           Operands &operands) {
    assert(init_param.input_count >= 2); // At least one one input tensor and axis
    assert(init_param.output_count == 1);

    // When there are N + 1 inputs, each input should be interpreted as follows:
    //
    //  [0, N) -> Input tensors
    //  N -> Axis
    //

    OperandIndexSequence inputs;
    for (uint32_t n = 0; n < init_param.input_count - 1; ++n)
    {
      inputs.append(OperandIndex{init_param.inputs[n]});
    }
    OperandIndexSequence outputs{init_param.outputs[0]};

    operation::Concat::Param param;
    const OperandIndex axis_index{init_param.inputs[init_param.input_count - 1]};
    param.axis = operands.at(axis_index).asScalar<int32_t>();

    return new operation::Concat{inputs, outputs, param};
  };

  _map[ANEURALNETWORKS_RESHAPE] = [](const OperationFactory::Param &init_param, Operands &) {
    assert(init_param.input_count == 2 && init_param.output_count == 1);

    // Each input should be interpreted as follows:
    //
    //  0 -> A tensor, specifying the tensor to be reshaped.
    //  1 -> A 1-D tensor of type ANEURALNETWORKS_TENSOR_INT32, defining the shape of the output
    //  tensor

    OperandIndexSequence inputs{init_param.inputs[0], init_param.inputs[1]};
    OperandIndexSequence outputs{init_param.outputs[0]};

    operation::Reshape::Param param{};

    return new operation::Reshape{inputs, outputs, param};
  };

  _map[ANEURALNETWORKS_FULLY_CONNECTED] = [](const OperationFactory::Param &init_param,
                                             Operands &operands) {
    assert(init_param.input_count == 4 && init_param.output_count == 1);

    // Each input should be interpreted as follows:
    //
    //  0 -> A tensor, specifying the input.
    //  1 -> A 2-D tensor, specifying the weights
    //  2 -> A 1-D tensor, specifying the bias
    //  3 -> An INT32 value, and has to be one of the FuseCode values

    OperandIndexSequence inputs{init_param.inputs[0], init_param.inputs[1], init_param.inputs[2]};
    OperandIndexSequence outputs{init_param.outputs[0]};

    operation::FullyConnected::Param param;
    const auto activation_index = OperandIndex{init_param.inputs[3]};
    param.activation =
      NNAPIConvert::getFusedActivation(operands.at(activation_index).asScalar<FuseCode>());
    param.weights_format = FullyConnectedWeightsFormat::Default;

    return new operation::FullyConnected{inputs, outputs, param};
  };

  _map[ANEURALNETWORKS_SOFTMAX] = [](const OperationFactory::Param &init_param,
                                     Operands &operands) {
    assert(init_param.input_count == 2 && init_param.output_count == 1);

    // Each input should be interpreted as follows:
    //
    //  0 -> A 2-D or 4-D tensor, specifying the tensor to be reshaped.
    //  1 ->  FLOAT32 value, specifying the positive scaling factor for the exponent, beta.

    OperandIndexSequence inputs{init_param.inputs[0]};
    OperandIndexSequence outputs{init_param.outputs[0]};

    const auto beta_index = OperandIndex{init_param.inputs[1]};

    operation::Softmax::Param param;
    param.beta = operands.at(beta_index).asScalar<float>();

    return new operation::Softmax{inputs, outputs, param};
  };

  _map[ANEURALNETWORKS_CAST] =
    getElementwiseUnaryGenerator(operation::ElementwiseUnary::Type::CAST);

  _map[ANEURALNETWORKS_CONV_2D] = [](const OperationFactory::Param &init_param,
                                     Operands &operands) {
    using operation::Conv2D;

    // inputCount is either 7 or 10 acccording to NN API specification.
    //  - Padding is implicit when inputCount is 7
    //  - Padding is explicit when inputCount is 10
    assert(init_param.input_count == 7 || init_param.input_count == 10 ||
           init_param.input_count == 13);
    assert(init_param.output_count == 1);

    //  0 -> IFM Tensor Index
    //  1 -> Kernel Tensor Index
    //  2 -> Bias Tensor Index

    OperandIndexSequence inputs{init_param.inputs[0], init_param.inputs[1], init_param.inputs[2]};
    OperandIndexSequence outputs{init_param.outputs[0]};

    Conv2D::Param param;
    if (init_param.input_count == 7) // support implicit padding
    {
      // Each input should be interpreted as follows:
      //
      //  3 -> Padding Code (ANEURALNETWORKS_PADDING_SAME or ANEURALNETWORKS_PADDING_VALID) Index
      //  4 -> Stride (width) Index
      //  5 -> Stride (height) INdex
      //  6 -> Activation Index

      const auto padding_index = OperandIndex{init_param.inputs[3]};
      const auto hstride_index = OperandIndex{init_param.inputs[4]};
      const auto vstride_index = OperandIndex{init_param.inputs[5]};
      const auto activation_index = OperandIndex{init_param.inputs[6]};

      param.padding.type =
        NNAPIConvert::getPaddingType(operands.at(padding_index).asScalar<PaddingCode>());
      param.stride = makeStride(operands, hstride_index, vstride_index);

      param.dilation.width_factor = 1;
      param.dilation.height_factor = 1;

      param.activation =
        NNAPIConvert::getFusedActivation(operands.at(activation_index).asScalar<FuseCode>());
    }
    else if (init_param.input_count == 10) // support explicit padding
    {
      // Each input should be interpreted as follows:
      //
      //  3 -> Padding_left index
      //  4 -> Padding_right index
      //  5 -> Padding_top index
      //  6 -> Padding_bottom index
      //  7 -> Stride (width) Index
      //  8 -> Stride (height) INdex
      //  9 -> Activation Index

      const auto padding_left_index = OperandIndex{init_param.inputs[3]};
      const auto padding_right_index = OperandIndex{init_param.inputs[4]};
      const auto padding_top_index = OperandIndex{init_param.inputs[5]};
      const auto padding_bottom_index = OperandIndex{init_param.inputs[6]};
      const auto hstride_index = OperandIndex{init_param.inputs[7]};
      const auto vstride_index = OperandIndex{init_param.inputs[8]};
      const auto activation_index = OperandIndex{init_param.inputs[9]};

      param.padding.type = PaddingType::EXPLICIT;
      param.padding.param = makeExplicitPadding(operands, padding_left_index, padding_right_index,
                                                padding_top_index, padding_bottom_index);
      param.stride = makeStride(operands, hstride_index, vstride_index);

      param.dilation.width_factor = 1;
      param.dilation.height_factor = 1;

      param.activation =
        NNAPIConvert::getFusedActivation(operands.at(activation_index).asScalar<FuseCode>());
    }
    else if (init_param.input_count == 13) // support dilation
    {
      // Each input should be interpreted as follows:
      //
      //  3 -> Padding_left Index
      //  4 -> Padding_right Index
      //  5 -> Padding_top Index
      //  6 -> Padding_bottom Index
      //  7 -> Stride (width) Index
      //  8 -> Stride (height) Index
      //  9 -> Activation Index
      //  11 -> Dilation (width_factor) Index
      //  12 -> Dilation (height_factor) INdex

      const auto padding_left_index = OperandIndex{init_param.inputs[3]};
      const auto padding_right_index = OperandIndex{init_param.inputs[4]};
      const auto padding_top_index = OperandIndex{init_param.inputs[5]};
      const auto padding_bottom_index = OperandIndex{init_param.inputs[6]};
      const auto hstride_index = OperandIndex{init_param.inputs[7]};
      const auto vstride_index = OperandIndex{init_param.inputs[8]};
      const auto activation_index = OperandIndex{init_param.inputs[9]};
      const auto width_factor_index = OperandIndex{init_param.inputs[11]};
      const auto height_factor_index = OperandIndex{init_param.inputs[12]};

      param.padding.type = PaddingType::EXPLICIT;
      param.padding.param = makeExplicitPadding(operands, padding_left_index, padding_right_index,
                                                padding_top_index, padding_bottom_index);
      param.stride = makeStride(operands, hstride_index, vstride_index);

      auto width_factor = operands.at(width_factor_index).asScalar<int32_t>();
      auto height_factor = operands.at(height_factor_index).asScalar<int32_t>();

      param.dilation.width_factor = width_factor;
      param.dilation.height_factor = height_factor;

      param.activation =
        NNAPIConvert::getFusedActivation(operands.at(activation_index).asScalar<FuseCode>());
    }
    else
    {
      throw std::runtime_error{"Conv2D: unsupported input operand count"};
    }

    return new Conv2D{inputs, outputs, param};
  };

  _map[ANEURALNETWORKS_ADD] =
    getBinaryArithmeticGenerator(onert::ir::operation::BinaryArithmetic::ArithmeticType::ADD);

  _map[ANEURALNETWORKS_ADDV2_EX] = _map[ANEURALNETWORKS_ADD];

  _map[ANEURALNETWORKS_REDUCE_SUM] =
    getReduceGenerator(onert::ir::operation::Reduce::ReduceType::SUM);

  _map[ANEURALNETWORKS_SUB] =
    getBinaryArithmeticGenerator(onert::ir::operation::BinaryArithmetic::ArithmeticType::SUB);

  _map[ANEURALNETWORKS_SLICE] = [](const OperationFactory::Param &init_param, Operands &) {
    assert(init_param.input_count == 3 && init_param.output_count == 1);

    OperandIndexSequence outputs{init_param.outputs[0]};

    // Each input should be interpreted as follows:
    //
    //  0 -> Input Tensor Index
    //  1 -> Begins Tensor Index
    //  2 -> Sizes Tensor Index
    OperandIndexSequence inputs{init_param.inputs[0], init_param.inputs[1], init_param.inputs[2]};

    return new operation::Slice{inputs, outputs};
  };

  _map[ANEURALNETWORKS_STRIDED_SLICE] = [](const OperationFactory::Param &init_param,
                                           Operands &operands) {
    assert(init_param.input_count == 7 && init_param.output_count == 1);

    OperandIndexSequence inputs{init_param.inputs[0], init_param.inputs[1], init_param.inputs[2],
                                init_param.inputs[3]};
    OperandIndexSequence outputs{init_param.outputs[0]};

    // Each input should be interpreted as follows:
    //
    //  1 -> A 1-D Tensor of {@link ANEURALNETWORKS_TENSOR_INT32}, the starts of
    //       the dimensions of the input tensor to be sliced. The length must be
    //       of rank(input0).
    //  2 -> A 1-D Tensor of {@link ANEURALNETWORKS_TENSOR_INT32}, the ends of
    //       the dimensions of the input tensor to be sliced. The length must be
    //       of rank(input0).
    //  3 -> A 1-D Tensor of {@link ANEURALNETWORKS_TENSOR_INT32}, the strides of
    //       the dimensions of the input tensor to be sliced. The length must be
    //       of rank(input0).
    //  4 -> An {@link ANEURALNETWORKS_INT32} scalar, begin_mask. If the ith bit
    //       of begin_mask is set, begin[i] is ignored and the fullest possible
    //       range in that dimension is used instead.
    //  5 -> An {@link ANEURALNETWORKS_INT32} scalar, end_mask. If the ith bit of
    //       end_mask is set, end[i] is ignored and the fullest possible range in
    //       that dimension is used instead.
    //  6 -> An {@link ANEURALNETWORKS_INT32} scalar, shrink_axis_mask. An int32
    //       mask. If the ith bit of shrink_axis_mask is set, it implies that the
    //       ith specification shrinks the dimensionality by 1. A slice of size 1
    //       starting from begin[i] in the dimension must be preserved.

    operation::StridedSlice::Param param;

    param.begin_mask = operands.at(OperandIndex{init_param.inputs[4]}).asScalar<std::int32_t>();
    param.end_mask = operands.at(OperandIndex{init_param.inputs[5]}).asScalar<std::int32_t>();
    param.shrink_axis_mask =
      operands.at(OperandIndex{init_param.inputs[6]}).asScalar<std::int32_t>();

    return new operation::StridedSlice{inputs, outputs, param};
  };

  _map[ANEURALNETWORKS_TRANSPOSE] = createSimpleBinaryOp<operation::Transpose>;

  _map[ANEURALNETWORKS_MUL] =
    getBinaryArithmeticGenerator(onert::ir::operation::BinaryArithmetic::ArithmeticType::MUL);

  _map[ANEURALNETWORKS_SQUEEZE] = [](const OperationFactory::Param &init_param,
                                     Operands &operands) {
    assert(init_param.input_count == 1 || init_param.input_count == 2);
    assert(init_param.output_count == 1);

    OperandIndexSequence outputs{init_param.outputs[0]};

    // Each input should be interpreted as follows:
    //
    // 0 -> An n-D tensor, the tensor to be squeezed.
    // 1 -> An optional 1-D tensor of ANEURALNETWORKS_TENSOR_INT32. The dimensions to squeeze.
    //      If specified only squeezes the dimensions listed. Otherwise, squeezes all dimensions.
    //      The dimension index starts at 0. An error must be reported if squeezing a dimension that
    //      is not 1.

    // Add mandatory input index
    OperandIndexSequence inputs{init_param.inputs[0]};

    // Add dims index if specified
    operation::Squeeze::Param param{};
    if (init_param.input_count == 2)
    {
      auto squeeze_dims_idx = OperandIndex{init_param.inputs[1]};
      assert(operands.at(squeeze_dims_idx).shape().rank() == 1);
      assert(operands.at(squeeze_dims_idx).shape().dim(0) >= 0);
      assert(static_cast<uint32_t>(operands.at(squeeze_dims_idx).shape().dim(0)) <=
             sizeof(param.dims));
      param.ndim = operands.at(squeeze_dims_idx).shape().dim(0);
      if (param.ndim > 0)
      {
        assert(operands.at(squeeze_dims_idx).data());
        memcpy(param.dims, operands.at(squeeze_dims_idx).data()->base(),
               param.ndim * sizeof(param.dims[0]));
      }
    }

    return new operation::Squeeze{inputs, outputs, param};
  };

  _map[ANEURALNETWORKS_TANH] = getElementwiseActivationGenerator(
    onert::ir::operation::ElementwiseActivation::Type::TANH, 1.f, 1.f);

  _map[ANEURALNETWORKS_LOG] = getElementwiseUnaryGenerator(operation::ElementwiseUnary::Type::LOG);

  _map[ANEURALNETWORKS_LOGISTIC] =
    getElementwiseActivationGenerator(onert::ir::operation::ElementwiseActivation::Type::LOGISTIC);

  _map[ANEURALNETWORKS_DIV] =
    getBinaryArithmeticGenerator(onert::ir::operation::BinaryArithmetic::ArithmeticType::DIV);

  _map[ANEURALNETWORKS_EXP] = getElementwiseUnaryGenerator(operation::ElementwiseUnary::Type::EXP);

  // Each input should be interpreted as follows:
  //  0 -> Input Tensor Index
  //  1 -> Axis Tensor Index
  _map[ANEURALNETWORKS_EXPAND_DIMS] = createSimpleBinaryOp<operation::ExpandDims>;

  _map[ANEURALNETWORKS_GREATER] =
    getComparisonGenerator(operation::Comparison::ComparisonType::Greater);
  _map[ANEURALNETWORKS_GREATER_EQUAL] =
    getComparisonGenerator(operation::Comparison::ComparisonType::GreaterEqual);
  _map[ANEURALNETWORKS_LESS] = getComparisonGenerator(operation::Comparison::ComparisonType::Less);
  _map[ANEURALNETWORKS_LESS_EQUAL] =
    getComparisonGenerator(operation::Comparison::ComparisonType::LessEqual);
  _map[ANEURALNETWORKS_NOT_EQUAL] =
    getComparisonGenerator(operation::Comparison::ComparisonType::NotEqual);
  _map[ANEURALNETWORKS_EQUAL] =
    getComparisonGenerator(operation::Comparison::ComparisonType::Equal);

  _map[ANEURALNETWORKS_REDUCE_ALL] =
    getReduceGenerator(onert::ir::operation::Reduce::ReduceType::ALL);

  _map[ANEURALNETWORKS_REDUCE_ANY] =
    getReduceGenerator(onert::ir::operation::Reduce::ReduceType::ANY);

  _map[ANEURALNETWORKS_REDUCE_MAX] =
    getReduceGenerator(onert::ir::operation::Reduce::ReduceType::MAX);

  _map[ANEURALNETWORKS_LOGICAL_AND] =
    getElementwiseBinaryGenerator(operation::ElementwiseBinary::ElementwiseBinaryType::LOGICAL_AND);

  _map[ANEURALNETWORKS_RSQRT] =
    getElementwiseUnaryGenerator(operation::ElementwiseUnary::Type::RSQRT);

  _map[ANEURALNETWORKS_SELECT] = [](const OperationFactory::Param &init_param, Operands &) {
    assert(init_param.input_count == 3 && init_param.output_count == 1);

    OperandIndexSequence outputs{init_param.outputs[0]};

    // Each input should be interpreted as follows:
    //
    //  0 -> Condition Tensor Index
    //  1 -> Input X(true) Tensor Index
    //  2 -> Input Y(false) Tensor Index
    OperandIndexSequence inputs{init_param.inputs[0], init_param.inputs[1], init_param.inputs[2]};

    return new operation::Select{inputs, outputs};
  };

  _map[ANEURALNETWORKS_SELECT_V2_EX] = _map[ANEURALNETWORKS_SELECT];

  _map[ANEURALNETWORKS_RELU] =
    getElementwiseActivationGenerator(onert::ir::operation::ElementwiseActivation::Type::RELU,
                                      onert::ir::operation::ElementwiseActivation::infinity, 0);

  _map[ANEURALNETWORKS_RESIZE_BILINEAR] = [](const OperationFactory::Param &init_param,
                                             Operands &operands) {
    assert(init_param.input_count == 3 && init_param.output_count == 1);

    OperandIndexSequence outputs{init_param.outputs[0]};

    // Each input should be interpreted as follows:
    //
    //  0 -> IFM Index
    //  1 -> Height Index
    //  2 -> Width Index
    OperandIndexSequence inputs{init_param.inputs[0]};

    operation::ResizeBilinear::Param param;
    param.height_out = operands.at(OperandIndex{init_param.inputs[1]}).asScalar<int32_t>();
    param.width_out = operands.at(OperandIndex{init_param.inputs[2]}).asScalar<int32_t>();
    param.align_corners = false;
    param.half_pixel_centers = false;
    return new operation::ResizeBilinear{inputs, outputs, param};
  };

  _map[ANEURALNETWORKS_RESIZE_NEAREST_NEIGHBOR] = [](const OperationFactory::Param &init_param,
                                                     Operands &operands) {
    assert((init_param.input_count == 3 || init_param.input_count == 4) &&
           init_param.output_count == 1);

    OperandIndexSequence outputs{init_param.outputs[0]};

    // Each input should be interpreted as follows:
    //
    //  0 -> IFM Index
    //  1 -> Height Index
    //  2 -> Width Index
    OperandIndexSequence inputs{init_param.inputs[0]};

    operation::ResizeNearestNeighbor::Param param;
    param.height_out = operands.at(OperandIndex{init_param.inputs[1]}).asScalar<int32_t>();
    param.width_out = operands.at(OperandIndex{init_param.inputs[2]}).asScalar<int32_t>();
    param.align_corners = false;
    // The layout input is not supported yet
    return new operation::ResizeNearestNeighbor{inputs, outputs, param};
  };

  _map[ANEURALNETWORKS_RELU1] = getElementwiseActivationGenerator(
    onert::ir::operation::ElementwiseActivation::Type::RELU, 1.f, -1.f);

  _map[ANEURALNETWORKS_RELU6] = getElementwiseActivationGenerator(
    onert::ir::operation::ElementwiseActivation::Type::RELU, 6.f, 0.f);

  _map[ANEURALNETWORKS_REVERSE_EX] = [](const OperationFactory::Param &init_param, Operands &) {
    assert(init_param.input_count == 2 && init_param.output_count == 1);

    // Each input should be interpreted as follows:
    //
    // 0 -> Input Tensor Index
    // 1 -> Axis Tensor Index

    OperandIndexSequence inputs{init_param.inputs[0], init_param.inputs[1]};
    OperandIndexSequence outputs{init_param.outputs[0]};

    return new operation::Reverse{inputs, outputs};
  };

  _map[ANEURALNETWORKS_RNN] = [](const OperationFactory::Param &init_param, Operands &operands) {
    assert(init_param.input_count == 6 && init_param.output_count == 2);

    // Each input should be interpreted as follows:
    //
    // 0 -> Input Tensor Index
    // 1 -> Weights Tensor Index
    // 2 -> Recurrent Weights Tensor Index
    // 3 -> Bias Tensor Index
    // 4 -> Hidden state (in) Index
    // 5 -> Activation Index

    OperandIndexSequence inputs;
    for (uint32_t n = 0; n < init_param.input_count - 1; ++n)
    {
      inputs.append(OperandIndex{init_param.inputs[n]});
    }
    OperandIndexSequence outputs;
    for (uint32_t n = 0; n < init_param.output_count; ++n)
    {
      outputs.append(OperandIndex{init_param.outputs[n]});
    }

    operation::RNN::Param param;
    const auto activation_index = OperandIndex{init_param.inputs[5]};
    param.activation =
      NNAPIConvert::getFusedActivation(operands.at(activation_index).asScalar<FuseCode>());

    return new operation::RNN{inputs, outputs, param};
  };

  _map[ANEURALNETWORKS_FLOOR] =
    getElementwiseUnaryGenerator(operation::ElementwiseUnary::Type::FLOOR);

  _map[ANEURALNETWORKS_SPACE_TO_BATCH_ND] = [](const OperationFactory::Param &init_param,
                                               Operands &) {
    assert(init_param.input_count == 3 && init_param.output_count == 1);

    OperandIndexSequence outputs{init_param.outputs[0]};

    // Each input should be interpreted as follows:
    //
    //  0 -> Input Tensor Index
    //  1 -> Block size Index
    //  2 -> Paddings Index
    OperandIndexSequence inputs;
    for (uint32_t n = 0; n < init_param.input_count; ++n)
    {
      inputs.append(OperandIndex{init_param.inputs[n]});
    }

    return new operation::SpaceToBatchND{inputs, outputs};
  };

  _map[ANEURALNETWORKS_SPACE_TO_DEPTH] = [](const OperationFactory::Param &init_param,
                                            Operands &operands) {
    assert(init_param.input_count == 2 && init_param.output_count == 1);

    OperandIndexSequence outputs{init_param.outputs[0]};

    // Each input should be interpreted as follows:
    //
    //  0 -> Input Tensor Index
    //  1 -> Block size Index
    OperandIndexSequence inputs{init_param.inputs[0]};

    operation::SpaceToDepth::Param param;
    param.block_size = operands.at(OperandIndex{init_param.inputs[1]}).asScalar<std::int32_t>();

    return new operation::SpaceToDepth{inputs, outputs, param};
  };

  _map[ANEURALNETWORKS_L2_POOL_2D] = getPool2DGenerator(operation::Pool2D::PoolType::L2);

  _map[ANEURALNETWORKS_EMBEDDING_LOOKUP] = [](const OperationFactory::Param &init_param,
                                              Operands &) {
    assert(init_param.input_count == 2 && init_param.output_count == 1);

    OperandIndexSequence outputs{init_param.outputs[0]};

    // Each input should be interpreted as follows:
    //
    //  0 -> Lookups Index
    //  1 -> Values Index
    OperandIndexSequence inputs{init_param.inputs[0], init_param.inputs[1]};

    return new operation::EmbeddingLookup{inputs, outputs};
  };

  _map[ANEURALNETWORKS_L2_NORMALIZATION] = [](const OperationFactory::Param &init_param,
                                              Operands &) {
    assert(init_param.input_count == 1 && init_param.output_count == 1);

    OperandIndexSequence outputs{init_param.outputs[0]};

    // Each input should be interpreted as follows:
    //  0 -> input Tensor Index
    OperandIndexSequence inputs{init_param.inputs[0]};

    return new operation::L2Normalization{inputs, outputs};
  };

  _map[ANEURALNETWORKS_HASHTABLE_LOOKUP] = [](const OperationFactory::Param &init_param,
                                              Operands &) {
    assert(init_param.input_count == 3 && init_param.output_count == 2);

    // Each output should be interpreted as follows:
    //
    //  0 -> Output Index
    //  1 -> Hits Index
    OperandIndexSequence outputs{init_param.outputs[0], init_param.outputs[1]};

    // Each input should be interpreted as follows:
    //
    //  0 -> Lookups Index
    //  1 -> Keys Index
    //  2 -> Values Index
    OperandIndexSequence inputs{init_param.inputs[0], init_param.inputs[1], init_param.inputs[2]};

    return new operation::HashtableLookup{inputs, outputs};
  };

  _map[ANEURALNETWORKS_PRELU] = [](const OperationFactory::Param &init_param, Operands &) {
    assert(init_param.input_count == 2 && init_param.output_count == 1);

    OperandIndexSequence outputs{init_param.outputs[0]};

    // Each input should be interpreted as follows:
    //
    //  0 -> input Tensor Index
    //  1 -> alpha Tensor Index
    OperandIndexSequence inputs{init_param.inputs[0], init_param.inputs[1]};

    return new operation::PReLU{inputs, outputs};
  };

  _map[ANEURALNETWORKS_TRANSPOSE_CONV_EX] = [](const OperationFactory::Param &init_param,
                                               Operands &operands) {
    assert(init_param.input_count == 6 && init_param.output_count == 1);

    OperandIndexSequence outputs{init_param.outputs[0]};

    // Each input should be interpreted as follows:
    //
    //  0 -> Output Shape Index
    //  1 -> Weights Index
    //  2 -> Input Tensor Index
    //  3 -> Padding Type
    //  4 -> Stride width
    //  5 -> Stride height

    OperandIndexSequence inputs{init_param.inputs[0], init_param.inputs[1], init_param.inputs[2]};

    operation::TransposeConv::Param param;

    const auto padding_index = OperandIndex{init_param.inputs[3]};
    const auto hstride_index = OperandIndex{init_param.inputs[4]};
    const auto vstride_index = OperandIndex{init_param.inputs[5]};

    param.padding.type =
      NNAPIConvert::getPaddingType(operands.at(padding_index).asScalar<PaddingCode>());
    param.stride = makeStride(operands, hstride_index, vstride_index);

    return new operation::TransposeConv{inputs, outputs, param};
  };

  _map[ANEURALNETWORKS_SQRT] =
    getElementwiseUnaryGenerator(operation::ElementwiseUnary::Type::SQRT);

  _map[ANEURALNETWORKS_LOGICAL_OR] =
    getElementwiseBinaryGenerator(operation::ElementwiseBinary::ElementwiseBinaryType::LOGICAL_OR);

  _map[ANEURALNETWORKS_LOGICAL_NOT] =
    getElementwiseUnaryGenerator(operation::ElementwiseUnary::Type::LOGICAL_NOT);

  _map[ANEURALNETWORKS_LSTM] = [](const OperationFactory::Param &init_param, Operands &operands) {
    assert(init_param.input_count == 23 && init_param.output_count == 4);

    // Each input should be interpreted as follows:
    //
    // 0 -> Input Tensor Index
    // 1 -> Input to Input Tensor Index
    // 2 -> Input to Forget Tensor Index
    // 3 -> Input to Cell Tensor Index
    // 4 -> Input to Output Tensor Index
    // 5 -> Recurrent to Input Weights Tensor Index
    // 6 -> Recurrent to Forget Weights Tensor Index
    // 7 -> Recurrent to Cell Weights Tensor Index
    // 8 -> Recurrent to Output Weights Tensor Index
    // 9 -> Cell to Input Weights Tensor Index
    // 10 -> Cell to Forget Weights Tensor Index
    // 11 -> Cell to Output Weights Tensor Index
    // 12 -> Input Gate Bias Tensor Index
    // 13 -> Forget Gate Bias Tensor Index
    // 14 -> Cell Bias Tensor Index
    // 15 -> Output Gate Bias Tensor Index
    // 16 -> Projection Weights Tensor Index
    // 17 -> Projection Bias Tensor Index
    // 18 -> Output State In Tensor Index
    // 19 -> Cell State In Tensor Index
    OperandIndexSequence inputs;
    for (uint32_t n = 0; n < init_param.input_count - 3; ++n)
    {
      inputs.append(OperandIndex{init_param.inputs[n]});
    }

    // Each output should be interpreted as follows:
    //
    // 0 -> Scratch Buffer Tensor Index
    // 1 -> Output State Out Tensor Index
    // 2 -> Cell State Out Tensor Index
    // 3 -> Output Tensor Index
    OperandIndexSequence outputs;
    for (uint32_t n = 0; n < init_param.output_count; ++n)
    {
      outputs.append(OperandIndex{init_param.outputs[n]});
    }

    operation::LSTM::Param param;
    param.activation = getActivation(operands, OperandIndex{init_param.inputs[20]});
    param.cell_threshold = operands.at(OperandIndex{init_param.inputs[21]}).asScalar<float>();
    param.projection_threshold = operands.at(OperandIndex{init_param.inputs[22]}).asScalar<float>();
    // This is initialization to prevent warning or error by static code analyzer. LSTM operation
    // does not need time_major
    param.time_major = false;

    return new operation::LSTM{inputs, outputs, param};
  };

  _map[ANEURALNETWORKS_UNIDIRECTIONAL_SEQUENCE_LSTM] = [](const OperationFactory::Param &init_param,
                                                          Operands &operands) {
    assert((init_param.input_count >= 24 || init_param.input_count <= 28) &&
           (init_param.output_count >= 1 && init_param.output_count <= 3));

    // Each input should be interpreted as follows:
    //
    // 0 -> Input Tensor Index
    // 1 -> Input to Input Tensor Index
    // 2 -> Input to Forget Tensor Index
    // 3 -> Input to Cell Tensor Index
    // 4 -> Input to Output Tensor Index
    // 5 -> Recurrent to Input Weights Tensor Index
    // 6 -> Recurrent to Forget Weights Tensor Index
    // 7 -> Recurrent to Cell Weights Tensor Index
    // 8 -> Recurrent to Output Weights Tensor Index
    // 9 -> Cell to Input Weights Tensor Index
    // 10 -> Cell to Forget Weights Tensor Index
    // 11 -> Cell to Output Weights Tensor Index
    // 12 -> Input Gate Bias Tensor Index
    // 13 -> Forget Gate Bias Tensor Index
    // 14 -> Cell Bias Tensor Index
    // 15 -> Output Gate Bias Tensor Index
    // 16 -> Projection Weights Tensor Index
    // 17 -> Projection Bias Tensor Index
    // 18 -> Output State In Tensor Index
    // 19 -> Cell State In Tensor Index
    assert(init_param.input_count - 3 > 20);
    OperandIndexSequence inputs;
    for (uint32_t n = 0; n < 20; ++n)
    {
      inputs.append(OperandIndex{init_param.inputs[n]});
    }

    // 24 -> Input Layer Normalization Weights Tensor Index
    // 25 -> Forget Layer Normalization Weights Tensor Index
    // 26 -> Cell Layer Normalization Weights Tensor Index
    // 27 -> Output Layer Normalization Weights Tensor Index
    if (init_param.input_count > 24)
    {
      for (uint32_t n = 24; n < 28; ++n)
      {
        if (init_param.input_count > n)
        {
          inputs.append(OperandIndex{init_param.inputs[n]});
        }
      }
    }

    // Each output should be interpreted as follows:
    //
    // 0 -> Output Tensor Index -> 3
    // 1 -> Output State Out Tensor Index
    // 2 -> Cell State Out Tensor Index
    const OperandIndex scratch_buffer_index;
    OperandIndex output_state_index =
      init_param.output_count >= 2 ? OperandIndex{init_param.outputs[1]} : OperandIndex();
    OperandIndex cell_state_index =
      init_param.output_count >= 3 ? OperandIndex{init_param.outputs[2]} : OperandIndex();
    const OperandIndex output_index = OperandIndex{init_param.outputs[0]};
    OperandIndexSequence outputs{scratch_buffer_index, output_state_index, cell_state_index,
                                 output_index};

    operation::LSTM::Param param;
    param.activation = getActivation(operands, OperandIndex{init_param.inputs[20]});
    param.cell_threshold = operands.at(OperandIndex{init_param.inputs[21]}).asScalar<float>();
    param.projection_threshold = operands.at(OperandIndex{init_param.inputs[22]}).asScalar<float>();
    param.time_major = operands.at(OperandIndex{init_param.inputs[23]}).asScalar<bool>();

    return new operation::LSTM{inputs, outputs, param};
  };

  _map[ANEURALNETWORKS_SQUARED_DIFFERENCE_EX] = [](const OperationFactory::Param &init_param,
                                                   Operands &) {
    assert(init_param.input_count == 2 && init_param.output_count == 1);

    OperandIndexSequence outputs{init_param.outputs[0]};

    // Each input should be interpreted as follows:
    //
    //  0 -> LHS Tensor Index
    //  1 -> RHS Tensor Index
    OperandIndexSequence inputs{init_param.inputs[0], init_param.inputs[1]};

    return new operation::SquaredDifference{inputs, outputs};
  };

  _map[ANEURALNETWORKS_TOPK_V2] = [](const OperationFactory::Param &init_param,
                                     Operands &operands) {
    assert(init_param.input_count == 2 && init_param.output_count == 2);

    // Each output should be interpreted as follows:
    //
    //  0 -> Index for Output Values
    //  1 -> Index for Output Indices
    OperandIndexSequence outputs{init_param.outputs[0], init_param.outputs[1]};

    // Each input should be interpreted as follows:
    //
    //  0 -> Index for Input Data
    //  1 -> Index for K
    OperandIndexSequence inputs{init_param.inputs[0]};

    operation::TopKV2::Param param;
    param.k = operands.at(OperandIndex{init_param.inputs[1]}).asScalar<std::int32_t>();

    return new operation::TopKV2{inputs, outputs, param};
  };

  _map[ANEURALNETWORKS_GATHER] = [](const OperationFactory::Param &init_param, Operands &operands) {
    assert(init_param.input_count == 3 && init_param.output_count == 1);

    OperandIndexSequence outputs{init_param.outputs[0]};

    // Each input should be interpreted as follows:
    //
    //  0 -> input Tensor Index
    //  1 -> axis Index
    //  2 -> indices Tensor Index
    OperandIndexSequence inputs{init_param.inputs[0], init_param.inputs[2]};

    operation::Gather::Param param;
    param.axis = operands.at(OperandIndex{init_param.inputs[1]}).asScalar<int32_t>();

    return new operation::Gather{inputs, outputs, param};
  };

  _map[ANEURALNETWORKS_NEG] = getElementwiseUnaryGenerator(operation::ElementwiseUnary::Type::NEG);

  _map[ANEURALNETWORKS_ABS] = getElementwiseUnaryGenerator(operation::ElementwiseUnary::Type::ABS);

  _map[ANEURALNETWORKS_ARGMAX] = [](const OperationFactory::Param &init_param, Operands &) {
    assert(init_param.input_count == 2 && init_param.output_count == 1);

    OperandIndexSequence outputs{init_param.outputs[0]};

    // Each input should be interpreted as follows:
    //
    //  0 -> Input Tensor Index
    //  1 -> Axis Tensor Index
    OperandIndexSequence inputs{init_param.inputs[0], init_param.inputs[1]};

    operation::ArgMinMax::Param param;
    // NNAPI ARGMAX output type is always int32
    param.output_type = DataType::INT32;
    param.is_arg_max = true;

    return new operation::ArgMinMax{inputs, outputs, param};
  };

  _map[ANEURALNETWORKS_ARGMIN] = [](const OperationFactory::Param &init_param, Operands &) {
    assert(init_param.input_count == 2 && init_param.output_count == 1);

    OperandIndexSequence outputs{init_param.outputs[0]};

    // Each input should be interpreted as follows:
    //
    //  0 -> Input Tensor Index
    //  1 -> Axis Tensor Index
    OperandIndexSequence inputs{init_param.inputs[0], init_param.inputs[1]};

    operation::ArgMinMax::Param param;
    // NNAPI ARGMIN output type is always int32
    param.output_type = DataType::INT32;
    param.is_arg_max = false;

    return new operation::ArgMinMax{inputs, outputs, param};
  };

  _map[ANEURALNETWORKS_DEQUANTIZE] =
    getElementwiseUnaryGenerator(operation::ElementwiseUnary::Type::DEQUANTIZE);

  _map[ANEURALNETWORKS_MEAN] = [](const OperationFactory::Param &init_param, Operands &operands) {
    assert(init_param.input_count == 3 && init_param.output_count == 1);

    OperandIndexSequence outputs{init_param.outputs[0]};

    // Each input should be interpreted as follows:
    //
    //  0 -> ifm Tensor Index
    //  1 -> axis Tensor Index
    //  2 -> keep_dims Index
    OperandIndexSequence inputs{init_param.inputs[0], init_param.inputs[1]};

    operation::Reduce::Param param;
    param.reduce_type = operation::Reduce::ReduceType::MEAN;
    param.keep_dims = operands.at(OperandIndex{init_param.inputs[2]}).asScalar<int32_t>() != 0;

    return new operation::Reduce{inputs, outputs, param};
  };

  _map[ANEURALNETWORKS_LOCAL_RESPONSE_NORMALIZATION] = [](const OperationFactory::Param &init_param,
                                                          Operands &operands) {
    assert(init_param.input_count == 5 && init_param.output_count == 1);

    OperandIndexSequence outputs{init_param.outputs[0]};

    OperandIndexSequence inputs{init_param.inputs[0]};

    operation::LocalResponseNormalization::Param param;
    param.radius = operands.at(OperandIndex{init_param.inputs[1]}).asScalar<std::int32_t>();
    param.bias = operands.at(OperandIndex{init_param.inputs[2]}).asScalar<float>();
    param.alpha = operands.at(OperandIndex{init_param.inputs[3]}).asScalar<float>();
    param.beta = operands.at(OperandIndex{init_param.inputs[4]}).asScalar<float>();

    return new operation::LocalResponseNormalization{inputs, outputs, param};
  };

  _map[ANEURALNETWORKS_DEPTH_TO_SPACE] = [](const OperationFactory::Param &init_param,
                                            Operands &operands) {
    assert(init_param.input_count == 2 && init_param.output_count == 1);

    OperandIndexSequence outputs{init_param.outputs[0]};

    // Each input should be interpreted as follows:
    //
    //  0 -> Input Tensor Index
    //  1 -> Block size Index
    OperandIndexSequence inputs{init_param.inputs[0]};

    operation::DepthToSpace::Param param;
    param.block_size = operands.at(OperandIndex{init_param.inputs[1]}).asScalar<std::int32_t>();

    return new operation::DepthToSpace{inputs, outputs, param};
  };

  _map[ANEURALNETWORKS_PACK_EX] = [](const OperationFactory::Param &init_param,
                                     Operands &operands) {
    assert(init_param.input_count >= 3 && init_param.output_count == 1);

    OperandIndexSequence outputs{init_param.outputs[0]};
    OperandIndexSequence inputs;
    for (uint32_t n = 0; n < init_param.input_count - 2; ++n)
    {
      inputs.append(OperandIndex{init_param.inputs[n]});
    }

    operation::Pack::Param param;
    const auto num_index = OperandIndex{init_param.inputs[init_param.input_count - 2]};
    const auto axis_index = OperandIndex{init_param.inputs[init_param.input_count - 1]};
    param.num = operands.at(num_index).asScalar<int32_t>();
    param.axis = operands.at(axis_index).asScalar<int32_t>();

    return new operation::Pack{inputs, outputs, param};
  };

  _map[ANEURALNETWORKS_REDUCE_MIN] =
    getReduceGenerator(onert::ir::operation::Reduce::ReduceType::MIN);

  _map[ANEURALNETWORKS_SPLIT] = [](const OperationFactory::Param &init_param, Operands &operands) {
    assert(init_param.input_count == 3);
    assert(init_param.output_count >= 1); // At least one output tensor and axis

    OperandIndexSequence inputs{init_param.inputs[1], init_param.inputs[0]};
    OperandIndexSequence outputs;
    for (uint32_t n = 0; n < init_param.output_count; ++n)
    {
      outputs.append(OperandIndex{init_param.outputs[n]});
    }

    operation::Split::Param param;
    param.num_splits = operands.at(OperandIndex{init_param.inputs[2]}).asScalar<std::int32_t>();

    return new operation::Split{inputs, outputs, param};
  };

  _map[ANEURALNETWORKS_SPLIT_V_EX] = [](const OperationFactory::Param &init_param,
                                        Operands &operands) {
    assert(init_param.input_count == 4);
    assert(init_param.output_count >= 1); // At least one output tensor and axis

    OperandIndexSequence inputs{init_param.inputs[0], init_param.inputs[1], init_param.inputs[2]};
    OperandIndexSequence outputs;
    for (uint32_t n = 0; n < init_param.output_count; ++n)
    {
      outputs.append(OperandIndex{init_param.outputs[n]});
    }

    operation::SplitV::Param param;
    param.num_splits = operands.at(OperandIndex{init_param.inputs[3]}).asScalar<std::int32_t>();
    return new operation::SplitV{inputs, outputs, param};
  };

  _map[ANEURALNETWORKS_UNPACK_EX] = [](const OperationFactory::Param &init_param,
                                       Operands &operands) {
    assert(init_param.input_count == 3 && init_param.output_count >= 1);

    OperandIndexSequence inputs{init_param.inputs[0]};
    OperandIndexSequence outputs;
    for (uint32_t n = 0; n < init_param.output_count; ++n)
    {
      outputs.append(OperandIndex{init_param.outputs[n]});
    }

    operation::Unpack::Param param;
    const auto num_index = OperandIndex{init_param.inputs[1]};
    const auto axis_index = OperandIndex{init_param.inputs[2]};
    param.num = operands.at(num_index).asScalar<int32_t>();
    param.axis = operands.at(axis_index).asScalar<int32_t>();

    return new operation::Unpack{inputs, outputs, param};
  };

  _map[ANEURALNETWORKS_PAD] = [](const OperationFactory::Param &init_param, Operands &) {
    assert(init_param.input_count >= 2 && init_param.input_count <= 3 &&
           init_param.output_count >= 1);

    OperandIndexSequence inputs{init_param.inputs[0], init_param.inputs[1]};
    if (init_param.input_count == 3)
    {
      inputs.append(OperandIndex{init_param.inputs[2]});
    }
    OperandIndexSequence outputs{init_param.outputs[0]};

    return new operation::Pad{inputs, outputs};
  };

  _map[ANEURALNETWORKS_PAD_V2] = _map[ANEURALNETWORKS_PAD];

  _map[ANEURALNETWORKS_MINIMUM] =
    getElementwiseBinaryGenerator(operation::ElementwiseBinary::ElementwiseBinaryType::MIN);

  _map[ANEURALNETWORKS_MAXIMUM] =
    getElementwiseBinaryGenerator(operation::ElementwiseBinary::ElementwiseBinaryType::MAX);

  _map[ANEURALNETWORKS_ONE_HOT_EX] = [](const OperationFactory::Param &init_param,
                                        Operands &operands) {
    assert(init_param.input_count == 5);
    assert(init_param.output_count == 1);
    // Each input should be interpreted as follows:
    //
    // 0 -> indices tensor
    // 1 -> depth tensor
    // 2 -> on_value tensor
    // 3 -> off_value tensor
    // 4 -> axis scalar
    OperandIndexSequence inputs;
    for (uint32_t n = 0; n < init_param.input_count - 1; ++n)
    {
      inputs.append(OperandIndex{init_param.inputs[n]});
    }
    OperandIndexSequence outputs{init_param.outputs[0]};

    operation::OneHot::Param param;
    param.axis = operands.at(OperandIndex{init_param.inputs[4]}).asScalar<std::int32_t>();

    return new operation::OneHot{inputs, outputs, param};
  };

  _map[ANEURALNETWORKS_COS_EX] =
    getElementwiseUnaryGenerator(operation::ElementwiseUnary::Type::COS);

  _map[ANEURALNETWORKS_SIN] = getElementwiseUnaryGenerator(operation::ElementwiseUnary::Type::SIN);

  _map[ANEURALNETWORKS_SHAPE_EX] = [](const OperationFactory::Param &init_param, Operands &) {
    assert(init_param.input_count == 1 && init_param.output_count == 1);

    OperandIndexSequence inputs{init_param.inputs[0]};
    OperandIndexSequence outputs{init_param.outputs[0]};

    return new operation::Shape{inputs, outputs};
  };

  _map[ANEURALNETWORKS_REDUCE_PROD] =
    getReduceGenerator(onert::ir::operation::Reduce::ReduceType::PROD);

  _map[ANEURALNETWORKS_ROUND_EX] =
    getElementwiseUnaryGenerator(operation::ElementwiseUnary::Type::ROUND);

  _map[ANEURALNETWORKS_RANGE_EX] = [](const OperationFactory::Param &init_param, Operands &) {
    assert(init_param.input_count == 3 && init_param.output_count == 1);

    OperandIndexSequence outputs{init_param.outputs[0]};

    // Each input should be interpreted as follows:
    //  0 -> start Tensor Index
    //  1 -> limit Tensor Index
    //  2 -> delta Tensor Index

    OperandIndexSequence inputs{init_param.inputs[0], init_param.inputs[1], init_param.inputs[2]};

    return new operation::Range{inputs, outputs};
  };

  // Each input should be interpreted as follows:
  //  0 -> LHS Tensor Index
  //  1 -> RHS Tensor Index
  _map[ANEURALNETWORKS_POW] = createSimpleBinaryOp<operation::Pow>;

  // Each input should be interpreted as follows:
  //  0 -> A tensor, specifying the input.
  //  1 -> A 1-D tensor, specifying the value
  _map[ANEURALNETWORKS_FILL_EX] = createSimpleBinaryOp<operation::Fill>;

  _map[ANEURALNETWORKS_ZEROS_LIKE_EX] =
    getElementwiseUnaryGenerator(operation::ElementwiseUnary::Type::ZEROS_LIKE);
  // Each input should be interpreted as follows:
  //  0 -> Input Tensor Index
  //  1 -> Multiple Tensor Index
  _map[ANEURALNETWORKS_TILE] = createSimpleBinaryOp<operation::Tile>;

  _map[ANEURALNETWORKS_MATRIX_BAND_PART_EX] = [](const OperationFactory::Param &init_param,
                                                 Operands &) {
    assert(init_param.input_count == 3);
    assert(init_param.output_count == 1);
    // Each input should be interpreted as follows:
    //
    // 0 -> A tensor, input
    // 1 -> A 0-D tensor, number of lower diagnonals to keep
    // 2 -> A 0-D tensor, number of upper diagnonals to keep
    OperandIndexSequence inputs{init_param.inputs[0], init_param.inputs[1], init_param.inputs[2]};
    OperandIndexSequence outputs{init_param.outputs[0]};

    return new operation::MatrixBandPart{inputs, outputs};
  };

  _map[ANEURALNETWORKS_BATCH_MATMUL_EX] = [](const OperationFactory::Param &init_param,
                                             Operands &operands) {
    assert(init_param.input_count == 4 && init_param.output_count == 1);

    OperandIndexSequence outputs{init_param.outputs[0]};

    // Each input should be interpreted as follows:
    //
    //  0 -> Lhs Tensor Index
    //  1 -> Rhs Tensor Index
    //  2 -> adj_x boolean scalar Index
    //  3 -> adj_y boolean scalar Index

    OperandIndexSequence inputs{init_param.inputs[0], init_param.inputs[1]};

    operation::BatchMatMul::Param param;
    param.adj_x = operands.at(OperandIndex{init_param.inputs[2]}).asScalar<bool>();
    param.adj_y = operands.at(OperandIndex{init_param.inputs[3]}).asScalar<bool>();

    return new operation::BatchMatMul{inputs, outputs, param};
  };

  _map[ANEURALNETWORKS_EINSUM_EX] = [](const OperationFactory::Param &init_param,
                                       Operands &operands) {
    // Each input should be interpreted as follows:
    //
    //  0....n - 1 -> n Input Tensors Index
    //  n -> equation
    assert(init_param.input_count >= 1 && init_param.output_count == 1);

    OperandIndexSequence inputs;
    for (uint32_t n = 0; n < init_param.input_count - 1; ++n)
    {
      inputs.append(OperandIndex{init_param.inputs[n]});
    }
    OperandIndexSequence outputs{init_param.outputs[0]};

    operation::Einsum::Param param;
    const OperandIndex equation_index{init_param.inputs[init_param.input_count - 1]};
    std::vector<char> equation_vector = operands.at(equation_index).asVector<char>();
    param.equation = std::string(equation_vector.begin(), equation_vector.end());

    return new operation::Einsum{inputs, outputs, param};
  };

  //  0 -> Input Tensor Index
  //  1 -> int32, int64, An 1-D int tensor Index
  _map[ANEURALNETWORKS_BROADCAST_TO_EX] = createSimpleBinaryOp<operation::BroadcastTo>;

  _map[ANEURALNETWORKS_STATELESS_RANDOM_UNIFORM_EX] = [](const OperationFactory::Param &init_param,
                                                         Operands &) {
    assert(init_param.input_count == 2 && init_param.output_count == 1);
    OperandIndexSequence outputs{init_param.outputs[0]};

    // Each input should be interpreted as follows:
    //
    //  0 -> Shape Tensor Index
    //  1 -> int32, int64, An 1-D int tensor Index

    OperandIndexSequence inputs{init_param.inputs[0], init_param.inputs[1]};

    return new operation::StatelessRandomUniform{inputs, outputs};
  };

  _map[ANEURALNETWORKS_FUSED_BATCH_NORM_V3_EX] = [](const OperationFactory::Param &init_param,
                                                    Operands &operands) {
    // Each input should be interpreted as follows:
    //
    //  0....4  -> 5 Input Tensors Index
    //  n-2     -> is_training
    //  n-1     -> data_format
    //  n       -> epsilon

    assert(init_param.input_count == 8 && init_param.output_count == 1);

    OperandIndexSequence inputs;
    for (uint32_t n = 0; n < init_param.input_count - 3; ++n)
    {
      inputs.append(OperandIndex{init_param.inputs[n]});
    }
    OperandIndexSequence outputs{init_param.outputs[0]};

    operation::FusedBatchNorm::Param param;
    const OperandIndex is_training_index{init_param.inputs[init_param.input_count - 3]};
    param.is_training = operands.at(is_training_index).asScalar<bool>();

    const OperandIndex data_format_index{init_param.inputs[init_param.input_count - 2]};
    std::vector<char> data_format_vector = operands.at(data_format_index).asVector<char>();
    param.data_format = std::string(data_format_vector.begin(), data_format_vector.end());

    const OperandIndex epsilon_index{init_param.inputs[init_param.input_count - 1]};
    param.epsilon = operands.at(epsilon_index).asScalar<float>();
    return new operation::FusedBatchNorm{inputs, outputs, param};
  };

  _map[ANEURALNETWORKS_LOG_SOFTMAX] = [](const OperationFactory::Param &init_param,
                                         Operands &operands) {
    assert(init_param.input_count == 3 && init_param.output_count == 1);

    // Each input should be interpreted as follows:
    //
    //  0 -> A tensor specifying the input logits.
    //  1 -> A scalar, specifying the positive scaling factor for the exponent, beta.
    //  2 -> An scalar specifying the axis to reduce across.

    OperandIndexSequence inputs{init_param.inputs[0]};
    OperandIndexSequence outputs{init_param.outputs[0]};

    const auto beta_index = OperandIndex{init_param.inputs[1]};
    const auto axis_index = OperandIndex{init_param.inputs[2]};

    operation::LogSoftmax::Param param;
    param.beta = operands.at(beta_index).asScalar<float>();
    param.axis = operands.at(axis_index).asScalar<int>();

    return new operation::LogSoftmax{inputs, outputs, param};
  };

  _map[ANEURALNETWORKS_QUANTIZE] =
    getElementwiseUnaryGenerator(operation::ElementwiseUnary::Type::QUANTIZE);
}

Operation *OperationFactory::create(ANeuralNetworksOperationType type,
                                    const OperationFactory::Param &param, Operands &operands)
{
  auto it = _map.find(type);
  if (it == _map.end())
  {
    throw std::runtime_error("Unsupported operation type: " + std::to_string(type));
  }
  return it->second(param, operands);
}
