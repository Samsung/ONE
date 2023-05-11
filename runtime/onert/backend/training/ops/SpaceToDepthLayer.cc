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

#include "SpaceToDepthLayer.h"

#include "OperationUtils.h"

#include <cker/operation/SpaceToDepth.h>

namespace onert
{
namespace backend
{
namespace training
{
namespace ops
{
SpaceToDepthLayer::SpaceToDepthLayer() : _input(nullptr), _block_size(0), _output(nullptr)
{
  // DO NOTHING
}

template <typename T> void SpaceToDepthLayer::spaceToDepth()
{

  nnfw::cker::SpaceToDepthParams params;
  params.block_size = _block_size;

  nnfw::cker::SpaceToDepth(params, getShape(_input), getBuffer<T>(_input), getShape(_output),
                           getBuffer<T>(_output));
}

void SpaceToDepthLayer::configure(const IPortableTensor *input, const int32_t block_size,
                                  IPortableTensor *output)
{
  _input = input;
  _block_size = block_size;
  _output = output;
}

void SpaceToDepthLayer::run()
{
  if (_input->data_type() == OperandType::FLOAT32)
  {
    spaceToDepth<float>();
  }
  else if (_input->data_type() == OperandType::QUANT_UINT8_ASYMM)
  {
    spaceToDepth<uint8_t>();
  }
  else
  {
    throw std::runtime_error{"SpaceToDepth: unsupported data type"};
  }
}

} // namespace ops
} // namespace training
} // namespace backend
} // namespace onert
