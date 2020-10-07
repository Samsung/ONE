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

#include "BinaryArithmeticLayer.h"

#include <cker/operation/BinaryArithmeticOps.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
{

namespace
{

template <nnfw::cker::BinaryArithmeticOpType arithmetic_type, typename T>
void eval(const IPortableTensor *lhs, const IPortableTensor *rhs, IPortableTensor *output,
          nnfw::cker::BinaryArithmeticOpParam op_params)
{
  const auto lhsShape = getTensorShape(lhs);
  const auto rhsShape = getTensorShape(rhs);
  const bool need_broadcast = nnfw::cker::ProcessBroadcastShapes(lhsShape, rhsShape, &op_params);
  if (need_broadcast)
  {
    nnfw::cker::BroadcastBinaryArithmeticOp<arithmetic_type>(
        op_params, lhsShape, reinterpret_cast<const T *>(lhs->buffer()), rhsShape,
        reinterpret_cast<const T *>(rhs->buffer()), getTensorShape(output),
        reinterpret_cast<T *>(output->buffer()));
    return;
  }

  nnfw::cker::BinaryArithmeticOp<arithmetic_type>(
      op_params, lhsShape, reinterpret_cast<const T *>(lhs->buffer()), rhsShape,
      reinterpret_cast<const T *>(rhs->buffer()), getTensorShape(output),
      reinterpret_cast<T *>(output->buffer()));
}

template <nnfw::cker::BinaryArithmeticOpType arithmetic_type>
std::function<void(const IPortableTensor *, const IPortableTensor *, IPortableTensor *)>
generateKernelGeneric(const IPortableTensor *lhs, const ir::Activation activation,
                      nnfw::cker::BinaryArithmeticOpParam op_params)
{
  switch (lhs->data_type())
  {
    case OperandType::FLOAT32:
    {
      float output_activation_min = 0, output_activation_max = 0;
      CalculateActivationRange(activation, &output_activation_min, &output_activation_max);
      op_params.float_activation_max = output_activation_max;
      op_params.float_activation_min = output_activation_min;
      return std::bind(&eval<arithmetic_type, float>, std::placeholders::_1, std::placeholders::_2,
                       std::placeholders::_3, op_params);
      break;
    }
    case OperandType::INT32:
    {
      int32_t output_activation_min = 0, output_activation_max = 0;
      CalculateActivationRange(activation, &output_activation_min, &output_activation_max);
      op_params.quantized_activation_max = output_activation_max;
      op_params.quantized_activation_min = output_activation_min;
      return std::bind(eval<arithmetic_type, int32_t>, std::placeholders::_1, std::placeholders::_2,
                       std::placeholders::_3, op_params);
      break;
    }
    default:
      throw std::runtime_error{"BinaryArithmetic(generic): Unsupported data type"};
  }
}

void setAddOrSubQuant8Params(const IPortableTensor *lhs, const IPortableTensor *rhs,
                             IPortableTensor *output, ir::Activation activation,
                             nnfw::cker::BinaryArithmeticOpParam *params)
{
  int32_t output_activation_min, output_activation_max;
  CalculateActivationRangeUint8(activation, output, &output_activation_min, &output_activation_max);
  nnfw::cker::BinaryArithmeticOpParam &op_params = *params;
  op_params.quantized_activation_max = output_activation_max;
  op_params.quantized_activation_min = output_activation_min;
  // Parameters for scaled quantized computation
  op_params.left_shift = 20;
  // Zero-points of input and output tensors
  op_params.input1_offset = -lhs->data_offset();
  op_params.input2_offset = -rhs->data_offset();
  op_params.output_offset = output->data_offset();
  assert((op_params.input1_offset >= 0) && (op_params.input1_offset <= 255));
  assert((op_params.input2_offset >= 0) && (op_params.input2_offset <= 255));
  assert((op_params.output_offset >= 0) && (op_params.output_offset <= 255));

  // Compute normalized scale for _lhs and _rhs values,
  // and represent in 32-bit fixed point
  const double norm_max_scale = 2 * std::max(lhs->data_scale(), rhs->data_scale());
  const double real_lhs_scale = lhs->data_scale() / norm_max_scale;
  const double real_rhs_scale = rhs->data_scale() / norm_max_scale;
  // output scale is used to normalize final result, so we invert the scale here
  const double real_output_scale =
      norm_max_scale / (output->data_scale() * (1 << op_params.left_shift));

  // Represent the scales as fixed int32_t multipliers, and int32_t shifts
  QuantizeMultiplier(real_lhs_scale, &op_params.input1_multiplier, &op_params.input1_shift);
  QuantizeMultiplier(real_rhs_scale, &op_params.input2_multiplier, &op_params.input2_shift);
  QuantizeMultiplier(real_output_scale, &op_params.output_multiplier, &op_params.output_shift);
}

void setMulQuant8Params(const IPortableTensor *lhs, const IPortableTensor *rhs,
                        IPortableTensor *output, ir::Activation activation,
                        nnfw::cker::BinaryArithmeticOpParam *params)
{
  int32_t output_activation_min, output_activation_max;
  CalculateActivationRangeUint8(activation, output, &output_activation_min, &output_activation_max);
  nnfw::cker::BinaryArithmeticOpParam &op_params = *params;

  op_params.quantized_activation_max = output_activation_max;
  op_params.quantized_activation_min = output_activation_min;
  op_params.input1_offset = -lhs->data_offset();
  op_params.input2_offset = -rhs->data_offset();
  op_params.output_offset = output->data_offset();

  double real_multiplier = lhs->data_scale() * rhs->data_scale() / output->data_scale();
  QuantizeMultiplier(real_multiplier, &op_params.output_multiplier, &op_params.output_shift);
}

} // namespace

void BinaryArithmeticLayer::configure(const IPortableTensor *lhs, const IPortableTensor *rhs,
                                      IPortableTensor *output, const ir::Activation activation,
                                      const ArithmeticType arithmetic_type)
{
  assert(lhs != nullptr);
  assert(rhs != nullptr);
  assert(output != nullptr);

  _lhs = lhs;
  _rhs = rhs;
  _output = output;

  nnfw::cker::BinaryArithmeticOpParam op_params;
  switch (arithmetic_type)
  {
    case ArithmeticType::kAdd:
      if (_lhs->data_type() == OperandType::QUANT_UINT8_ASYMM)
      {
        setAddOrSubQuant8Params(_lhs, _rhs, _output, activation, &op_params);
        _kernel = std::bind(&eval<nnfw::cker::BinaryArithmeticOpType::ADD, uint8_t>,
                            std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                            op_params);
      }
      else
      {
        _kernel = generateKernelGeneric<nnfw::cker::BinaryArithmeticOpType::ADD>(_lhs, activation,
                                                                                 op_params);
      }
      break;
    case ArithmeticType::kSub:
      if (_lhs->data_type() == OperandType::QUANT_UINT8_ASYMM)
      {
        setAddOrSubQuant8Params(_lhs, _rhs, _output, activation, &op_params);
        op_params.input2_multiplier *= -1;
        _kernel = std::bind(&eval<nnfw::cker::BinaryArithmeticOpType::SUB, uint8_t>,
                            std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                            op_params);
      }
      else
      {
        _kernel = generateKernelGeneric<nnfw::cker::BinaryArithmeticOpType::SUB>(_lhs, activation,
                                                                                 op_params);
      }
      break;
    case ArithmeticType::kMul:
      if (_lhs->data_type() == OperandType::QUANT_UINT8_ASYMM)
      {
        nnfw::cker::BinaryArithmeticOpParam op_params;
        setMulQuant8Params(_lhs, _rhs, _output, activation, &op_params);
        _kernel = std::bind(&eval<nnfw::cker::BinaryArithmeticOpType::MUL, uint8_t>,
                            std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                            op_params);
      }
      else
      {
        _kernel = generateKernelGeneric<nnfw::cker::BinaryArithmeticOpType::MUL>(_lhs, activation,
                                                                                 op_params);
      }
      break;
    case ArithmeticType::kDiv:
      if (_lhs->data_type() == OperandType::QUANT_UINT8_ASYMM)
      {
        throw std::runtime_error{
            "BinaryArithmetic(Div): Div operation does not support quantization"};
      }
      else if (_lhs->data_type() == OperandType::INT32)
      {
        throw std::runtime_error{"BinaryArithmetic(Div): Unsupported data type"};
      }
      else
      {
        _kernel = generateKernelGeneric<nnfw::cker::BinaryArithmeticOpType::DIV>(_lhs, activation,
                                                                                 op_params);
      }
      break;
    default:
      throw std::runtime_error{"BinaryArithmetic: Unsupported BinaryArithmetic type"};
  }
}

void BinaryArithmeticLayer::run() { _kernel(_lhs, _rhs, _output); }

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert
