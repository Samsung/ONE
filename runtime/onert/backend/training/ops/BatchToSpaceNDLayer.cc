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

#include "BatchToSpaceNDLayer.h"

#include <cker/operation/BatchToSpaceND.h>

namespace onert
{
namespace backend
{
namespace training
{
namespace ops
{

BatchToSpaceNDLayer::BatchToSpaceNDLayer()
  : _input(nullptr), _output(nullptr), _block_shape(nullptr), _crops(nullptr)
{
  // DO NOTHING
}

template <typename T> void BatchToSpaceNDLayer::batchToSpaceNDGeneric()
{
  const int32_t NNapiCrops[]{0, 0, 0, 0};
  const int32_t *_crops_buffer;

  if (_crops == nullptr)
  {
    _crops_buffer = NNapiCrops;
  }
  else
  {
    _crops_buffer = getBuffer<int32_t>(_crops);
  }
  nnfw::cker::BatchToSpaceND<T>(getShape(_input), getBuffer<T>(_input),
                                getBuffer<int32_t>(_block_shape), _crops_buffer, getShape(_output),
                                getBuffer<T>(_output));
}

void BatchToSpaceNDLayer::configure(const IPortableTensor *input, IPortableTensor *output,
                                    IPortableTensor *block_shape, IPortableTensor *crops)
{
  _output = output;
  _input = input;
  _block_shape = block_shape;
  _crops = crops;
}

void BatchToSpaceNDLayer::run()
{
  if (_output->data_type() == OperandType::FLOAT32)
  {
    batchToSpaceNDGeneric<float>();
  }
  else if (_output->data_type() == OperandType::QUANT_UINT8_ASYMM)
  {
    batchToSpaceNDGeneric<uint8_t>();
  }
  else
  {
    throw std::runtime_error{"NYI"};
  }
}

} // namespace ops
} // namespace training
} // namespace backend
} // namespace onert
