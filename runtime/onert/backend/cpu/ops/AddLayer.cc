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

#include "AddLayer.h"

#include <cker/operation/BinaryArithmeticOps.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
{

void AddLayer::addFloat32()
{
  float output_activation_min, output_activation_max;
  CalculateActivationRangeFloat(_activation, &output_activation_min, &output_activation_max);
  nnfw::cker::BinaryArithmeticOpParam op_params;
  op_params.type = nnfw::cker::BinaryArithmeticOpType::ADD;
  op_params.float_activation_max = output_activation_max;
  op_params.float_activation_min = output_activation_min;

  const bool need_broadcast =
      nnfw::cker::ProcessBroadcastShapes(getTensorShape(_lhs), getTensorShape(_rhs), &op_params);
  if (need_broadcast)
  {
    nnfw::cker::BroadcastBinaryArithmeticOp(
        op_params, getTensorShape(_lhs), reinterpret_cast<const float *>(_lhs->buffer()),
        getTensorShape(_rhs), reinterpret_cast<const float *>(_rhs->buffer()),
        getTensorShape(_output), reinterpret_cast<float *>(_output->buffer()));
    return;
  }

  nnfw::cker::BinaryArithmeticOp(
      op_params, getTensorShape(_lhs), reinterpret_cast<const float *>(_lhs->buffer()),
      getTensorShape(_rhs), reinterpret_cast<const float *>(_rhs->buffer()),
      getTensorShape(_output), reinterpret_cast<float *>(_output->buffer()));
}

void AddLayer::addInt32()
{
  int32_t output_activation_min, output_activation_max;
  CalculateActivationRangeInt32(_activation, &output_activation_min, &output_activation_max);
  nnfw::cker::BinaryArithmeticOpParam op_params;
  op_params.type = nnfw::cker::BinaryArithmeticOpType::ADD;
  op_params.quantized_activation_max = output_activation_max;
  op_params.quantized_activation_min = output_activation_min;

  const bool need_broadcast =
      nnfw::cker::ProcessBroadcastShapes(getTensorShape(_lhs), getTensorShape(_rhs), &op_params);
  if (need_broadcast)
  {
    nnfw::cker::BroadcastBinaryArithmeticOp(
        op_params, getTensorShape(_lhs), reinterpret_cast<const int32_t *>(_lhs->buffer()),
        getTensorShape(_rhs), reinterpret_cast<const int32_t *>(_rhs->buffer()),
        getTensorShape(_output), reinterpret_cast<int32_t *>(_output->buffer()));
    return;
  }

  nnfw::cker::BinaryArithmeticOp(
      op_params, getTensorShape(_lhs), reinterpret_cast<const int32_t *>(_lhs->buffer()),
      getTensorShape(_rhs), reinterpret_cast<const int32_t *>(_rhs->buffer()),
      getTensorShape(_output), reinterpret_cast<int32_t *>(_output->buffer()));
}

void AddLayer::addQuant8()
{
  int32_t output_activation_min, output_activation_max;
  CalculateActivationRangeUint8(_activation, _output, &output_activation_min,
                                &output_activation_max);
  // nnfw::cker::BinaryArithmeticOpParam op_params;
  // op_params.quantized_activation_max = output_activation_max;
  // op_params.quantized_activation_min = output_activation_min;

  // cker quant8 add is not implemented yet
  throw std::runtime_error{"NYI"};
}

void AddLayer::configure(const Tensor *lhs, const Tensor *rhs, const ir::Activation activation,
                         Tensor *output)
{
  assert(lhs != nullptr);
  assert(rhs != nullptr);
  assert(output != nullptr);

  _lhs = lhs;
  _rhs = rhs;
  _activation = activation;
  _output = output;
}

void AddLayer::run()
{
  if (_lhs->data_type() == OperandType::FLOAT32)
  {
    addFloat32();
  }
  else if (_lhs->data_type() == OperandType::QUANT_UINT8_ASYMM)
  {
    addQuant8();
  }
  else if (_output->data_type() == OperandType::INT32)
  {
    addInt32();
  }
  else
  {
    throw std::runtime_error{"Add: unsupported data type"};
  }
}

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert
