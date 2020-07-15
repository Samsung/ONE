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

#include "L2NormLayer.h"

#include "OperationUtils.h"

#include <cker/operation/L2Normalize.h>
#include <cker/Types.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
{

void L2NormLayer::configure(const IPortableTensor *input, IPortableTensor *output)
{
  assert(input != nullptr);
  assert(output != nullptr);

  _input = input;
  _output = output;
}

void L2NormLayer::run()
{
  switch (_input->data_type())
  {
    case OperandType::FLOAT32:
      nnfw::cker::L2NormalizeFloat32(
          getTensorShape(_input), reinterpret_cast<const float *>(_input->buffer()),
          getTensorShape(_output), reinterpret_cast<float *>(_output->buffer()));
      break;

    case OperandType::QUANT_UINT8_ASYMM:
    {
      nnfw::cker::L2NormParams params;
      assert(_input->data_offset() == 128);
      params.input_zero_point = _input->data_offset();
      nnfw::cker::L2NormalizeQuant8(
          params, getTensorShape(_input), reinterpret_cast<const uint8_t *>(_input->buffer()),
          getTensorShape(_output), reinterpret_cast<uint8_t *>(_output->buffer()));
    }
    break;

    default:
      throw std::runtime_error{"L2Norm: Unsupported data type"};
  }
}

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert
