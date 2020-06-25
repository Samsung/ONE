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

#include "GatherLayer.h"

#include "OperationUtils.h"

#include <cker/operation/Gather.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
{

void GatherLayer::configure(const IPortableTensor *input, const IPortableTensor *indices,
                            IPortableTensor *output, int32_t axis)
{
  _input = input;
  _indices = indices;
  _axis = axis;
  _output = output;
}

void GatherLayer::run()
{
  nnfw::cker::GatherParams op_params;
  op_params.axis = _axis;

  switch (_input->data_type())
  {
    case OperandType::FLOAT32:
      nnfw::cker::Gather<float>(
          op_params, getTensorShape(_input), reinterpret_cast<const float *>(_input->buffer()),
          getTensorShape(_indices), reinterpret_cast<const int32_t *>(_indices->buffer()),
          getTensorShape(_output), reinterpret_cast<float *>(_output->buffer()));
      break;
    case OperandType::QUANT_UINT8_ASYMM:
      nnfw::cker::Gather<uint8_t>(
          op_params, getTensorShape(_input), reinterpret_cast<const uint8_t *>(_input->buffer()),
          getTensorShape(_indices), reinterpret_cast<const int32_t *>(_indices->buffer()),
          getTensorShape(_output), reinterpret_cast<uint8_t *>(_output->buffer()));
      break;
    case OperandType::INT32:
      nnfw::cker::Gather<int32_t>(
          op_params, getTensorShape(_input), reinterpret_cast<const int32_t *>(_input->buffer()),
          getTensorShape(_indices), reinterpret_cast<const int32_t *>(_indices->buffer()),
          getTensorShape(_output), reinterpret_cast<int32_t *>(_output->buffer()));
      break;
    default:
      throw std::runtime_error("Gather: unsupported data type");
  }
}

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert
