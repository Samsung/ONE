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

#include "OperationUtils.h"

#include <cker/operation/BatchToSpaceND.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace kernel
{

BatchToSpaceNDLayer::BatchToSpaceNDLayer() : _input(nullptr), _output(nullptr)
{
  // DO NOTHING
}

void BatchToSpaceNDLayer::batchToSpaceNDFloat32()
{
  // nnfw::cker::BatchToSpaceNDParams opParams{ _block_shape->buffer(), _crops->buffer() };
  nnfw::cker::BatchToSpaceND(
      convertTensorToCkerShape(_input), reinterpret_cast<const float *>(_input->buffer()),
      convertTensorToCkerShape(_block_shape), reinterpret_cast<const int *>(_block_shape->buffer()),
      convertTensorToCkerShape(_crops), reinterpret_cast<const int *>(_crops->buffer()),
      convertTensorToCkerShape(_output), reinterpret_cast<float *>(_output->buffer()));
}

void BatchToSpaceNDLayer::batchToSpaceNDQuant8()
{
  // throw std::runtime_error{"NYT"};
}

void BatchToSpaceNDLayer::configure(const operand::Tensor *input, operand::Tensor *output,
                                    operand::Tensor *block_shape, operand::Tensor *crops)
{
  _output = output;
  _input = input;
  _block_shape = block_shape;
  _crops = crops;
}

void BatchToSpaceNDLayer::run()
{
  if (_input->data_type() == OperandType::FLOAT32)
  {
    batchToSpaceNDFloat32();
  }
  // else if (_input->data_type() == OperandType::QUANT8_ASYMM)
  // {
  //   batchToSpaceNDQuant8();
  // }
  else
  {
    throw std::runtime_error{"NYI"};
  }
}

} // namespace kernel
} // namespace cpu
} // namespace backend
} // namespace onert
