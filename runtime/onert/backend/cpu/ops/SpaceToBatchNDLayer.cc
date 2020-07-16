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

#include "SpaceToBatchNDLayer.h"

#include "OperationUtils.h"

#include <cker/operation/SpaceToBatchND.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
{
SpaceToBatchNDLayer::SpaceToBatchNDLayer()
    : _input(nullptr), _block_shape(nullptr), _padding(nullptr), _output(nullptr)
{
  // DO NOTHING
}

template <> uint32_t SpaceToBatchNDLayer::getPad<float>() { return 0; }
template <> uint32_t SpaceToBatchNDLayer::getPad<uint8_t>() { return _output->data_offset(); }

template <typename T> void SpaceToBatchNDLayer::spaceToBatchND()
{
  nnfw::cker::SpaceToBatchParams params;
  params.output_offset = getPad<T>();

  nnfw::cker::SpaceToBatchND(
      params, getTensorShape(_input), reinterpret_cast<const T *>(_input->buffer()),
      getTensorShape(_block_shape), reinterpret_cast<const int32_t *>(_block_shape->buffer()),
      getTensorShape(_padding), reinterpret_cast<const int32_t *>(_padding->buffer()),
      getTensorShape(_output), reinterpret_cast<T *>(_output->buffer()));
}

void SpaceToBatchNDLayer::configure(const IPortableTensor *input,
                                    const IPortableTensor *block_shape,
                                    const IPortableTensor *padding, IPortableTensor *output)
{
  _input = input;
  _block_shape = block_shape;
  _padding = padding;
  _output = output;
}

void SpaceToBatchNDLayer::run()
{
  if (_input->data_type() == OperandType::FLOAT32)
  {
    spaceToBatchND<float>();
  }
  else if (_input->data_type() == OperandType::QUANT_UINT8_ASYMM)
  {
    spaceToBatchND<uint8_t>();
  }
  else
  {
    throw std::runtime_error{"SpaceToBatchND: unsupported data type"};
  }
}

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert
