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

#include "SubLayer.h"

#include <cker/operation/BinaryArithmeticOps.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
{

void SubLayer::subFloat32()
{
  float output_activation_min = 0, output_activation_max = 0;
  CalculateActivationRange(_activation, &output_activation_min, &output_activation_max);
  nnfw::cker::BinaryArithmeticOpParam op_params;
  op_params.float_activation_max = output_activation_max;
  op_params.float_activation_min = output_activation_min;

  const bool need_broadcast =
      nnfw::cker::ProcessBroadcastShapes(getTensorShape(_lhs), getTensorShape(_rhs), &op_params);
  if (need_broadcast)
  {
    nnfw::cker::BroadcastBinaryArithmeticOp<nnfw::cker::BinaryArithmeticOpType::SUB>(
        op_params, getTensorShape(_lhs), reinterpret_cast<const float *>(_lhs->buffer()),
        getTensorShape(_rhs), reinterpret_cast<const float *>(_rhs->buffer()),
        getTensorShape(_output), reinterpret_cast<float *>(_output->buffer()));
    return;
  }

  nnfw::cker::BinaryArithmeticOp<nnfw::cker::BinaryArithmeticOpType::SUB>(
      op_params, getTensorShape(_lhs), reinterpret_cast<const float *>(_lhs->buffer()),
      getTensorShape(_rhs), reinterpret_cast<const float *>(_rhs->buffer()),
      getTensorShape(_output), reinterpret_cast<float *>(_output->buffer()));
}

void SubLayer::subQuant8()
{
  int32_t output_activation_min, output_activation_max;
  CalculateActivationRangeUint8(_activation, _output, &output_activation_min,
                                &output_activation_max);
  nnfw::cker::BinaryArithmeticOpParam op_params;
  op_params.quantized_activation_max = output_activation_max;
  op_params.quantized_activation_min = output_activation_min;
  // Parameters for scaled quantized computation
  op_params.left_shift = 20;
  // Zero-points of input and output tensors
  op_params.input1_offset = -_lhs->data_offset();
  op_params.input2_offset = -_rhs->data_offset();
  op_params.output_offset = _output->data_offset();
  assert((op_params.input1_offset >= 0) && (op_params.input1_offset <= 255));
  assert((op_params.input2_offset >= 0) && (op_params.input2_offset <= 255));
  assert((op_params.output_offset >= 0) && (op_params.output_offset <= 255));

  // Compute normalized scale for _lhs and _rhs values,
  // and represent in 32-bit fixed point
  const double norm_max_scale = 2 * std::max(_lhs->data_scale(), _rhs->data_scale());
  const double real_lhs_scale = _lhs->data_scale() / norm_max_scale;
  const double real_rhs_scale = _rhs->data_scale() / norm_max_scale;
  // output scale is used to normalize final result, so we invert the scale here
  const double real_output_scale =
      norm_max_scale / (_output->data_scale() * (1 << op_params.left_shift));

  // Represent the scales as fixed int32_t multipliers, and int32_t shifts
  QuantizeMultiplier(real_lhs_scale, &op_params.input1_multiplier, &op_params.input1_shift);
  QuantizeMultiplier(real_rhs_scale, &op_params.input2_multiplier, &op_params.input2_shift);
  op_params.input2_multiplier *= -1;
  QuantizeMultiplier(real_output_scale, &op_params.output_multiplier, &op_params.output_shift);

  const bool need_broadcast =
      nnfw::cker::ProcessBroadcastShapes(getTensorShape(_lhs), getTensorShape(_rhs), &op_params);
  if (need_broadcast)
  {
    nnfw::cker::BroadcastBinaryArithmeticOp<nnfw::cker::BinaryArithmeticOpType::SUB>(
        op_params, getTensorShape(_lhs), reinterpret_cast<const uint8_t *>(_lhs->buffer()),
        getTensorShape(_rhs), reinterpret_cast<const uint8_t *>(_rhs->buffer()),
        getTensorShape(_output), reinterpret_cast<uint8_t *>(_output->buffer()));
    return;
  }

  nnfw::cker::BinaryArithmeticOp<nnfw::cker::BinaryArithmeticOpType::SUB>(
      op_params, getTensorShape(_lhs), reinterpret_cast<const uint8_t *>(_lhs->buffer()),
      getTensorShape(_rhs), reinterpret_cast<const uint8_t *>(_rhs->buffer()),
      getTensorShape(_output), reinterpret_cast<uint8_t *>(_output->buffer()));
}

void SubLayer::configure(const IPortableTensor *lhs, const IPortableTensor *rhs,
                         const ir::Activation activation, IPortableTensor *output)
{
  _lhs = lhs;
  _rhs = rhs;
  _activation = activation;
  _output = output;
}

void SubLayer::run()
{
  if (_output->data_type() == OperandType::FLOAT32)
  {
    subFloat32();
  }
  else if (_output->data_type() == OperandType::QUANT_UINT8_ASYMM)
  {
    subQuant8();
  }
  else
  {
    throw std::runtime_error{"Sub: unsupported data type"};
  }
}

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert
