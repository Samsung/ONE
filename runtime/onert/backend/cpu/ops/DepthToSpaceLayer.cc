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

#include "DepthToSpaceLayer.h"

#include "OperationUtils.h"

#include <cker/operation/DepthToSpace.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
{
DepthToSpaceLayer::DepthToSpaceLayer() : _input(nullptr), _block_size(0), _output(nullptr)
{
  // DO NOTHING
}

template <typename T> void DepthToSpaceLayer::depthToSpace()
{
  nnfw::cker::DepthToSpace(getTensorShape(_input), reinterpret_cast<const T *>(_input->buffer()),
                           getTensorShape(_output), reinterpret_cast<T *>(_output->buffer()),
                           _block_size);
}

void DepthToSpaceLayer::configure(const IPortableTensor *input, const int32_t block_size,
                                  IPortableTensor *output)
{
  _input = input;
  _block_size = block_size;
  _output = output;
}

void DepthToSpaceLayer::run()
{
  if (_input->data_type() == OperandType::FLOAT32)
  {
    depthToSpace<float>();
  }
  else if (_input->data_type() == OperandType::INT32)
  {
    depthToSpace<int32_t>();
  }
  else if (_input->data_type() == OperandType::INT64)
  {
    depthToSpace<int64_t>();
  }
  else if (_input->data_type() == OperandType::QUANT_UINT8_ASYMM)
  {
    depthToSpace<uint8_t>();
  }
  else if (_input->data_type() == OperandType::QUANT_INT8_ASYMM)
  {
    depthToSpace<int8_t>();
  }
  else
  {
    throw std::runtime_error{"DepthToSpace: unsupported data type"};
  }
}

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert
