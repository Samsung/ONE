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

#include "FillLayer.h"

#include "OperationUtils.h"

#include <cker/operation/Fill.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
{

FillLayer::FillLayer() : _input(nullptr), _value(nullptr), _output(nullptr)
{
  // DO NOTHING
}

void FillLayer::configure(const IPortableTensor *input, const IPortableTensor *value,
                          IPortableTensor *output)
{
  _input = input;
  _value = value;
  _output = output;
}

void FillLayer::run()
{
  switch (_output->data_type())
  {
    case OperandType::FLOAT32:
      nnfw::cker::Fill<float *>(getTensorShape(_input), reinterpret_cast<int *>(_input->buffer()),
                                reinterpret_cast<float *>(_value->buffer()),
                                getTensorShape(_output),
                                reinterpret_cast<float *>(_output->buffer()));
      break;
    case OperandType::INT32:
      nnfw::cker::Fill<int32_t *>(getTensorShape(_input), reinterpret_cast<int *>(_input->buffer()),
                                  reinterpret_cast<int32_t *>(_value->buffer()),
                                  getTensorShape(_output),
                                  reinterpret_cast<int32_t *>(_output->buffer()));
      break;
    case OperandType::UINT32:
      nnfw::cker::Fill<uint32_t *>(
          getTensorShape(_input), reinterpret_cast<int *>(_input->buffer()),
          reinterpret_cast<uint32_t *>(_value->buffer()), getTensorShape(_output),
          reinterpret_cast<uint32_t *>(_output->buffer()));
      break;
    default:
      throw std::runtime_error{"Fill: unsupported data type"};
  }
}

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert
