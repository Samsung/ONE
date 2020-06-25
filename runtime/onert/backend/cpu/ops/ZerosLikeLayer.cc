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

#include "ZerosLikeLayer.h"

#include "OperationUtils.h"

namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
{
ZerosLikeLayer::ZerosLikeLayer() : _input(nullptr), _output(nullptr)
{
  // DO NOTHING
}

void ZerosLikeLayer::configure(const IPortableTensor *input, IPortableTensor *output)
{
  _input = input;
  _output = output;
}

void ZerosLikeLayer::run()
{
  if (!HaveSameShapes(_input, _output))
    throw std::runtime_error{"ZerosLike: input and output shape don't match."};

  auto element_size = getTensorShape(_input).FlatSize();

  switch (_input->data_type())
  {
    case OperandType::FLOAT32:
      memset(reinterpret_cast<float *>(_output->buffer()), 0, element_size * sizeof(float));
      break;
    case OperandType::INT32:
      memset(reinterpret_cast<int32_t *>(_output->buffer()), 0, element_size * sizeof(int32_t));
      break;
    default:
      throw std::runtime_error{"ZerosLike: unsupported data type"};
  }
}

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert
