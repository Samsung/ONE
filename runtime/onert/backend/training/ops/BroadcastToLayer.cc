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

#include "BroadcastToLayer.h"

#include <cker/operation/BroadcastTo.h>

namespace onert
{
namespace backend
{
namespace training
{
namespace ops
{

BroadcastToLayer::BroadcastToLayer() : _input(nullptr), _shape(nullptr), _output(nullptr)
{
  // DO NOTHING
}

void BroadcastToLayer::configure(const IPortableTensor *input, const IPortableTensor *shape,
                                 IPortableTensor *output)
{
  _input = input;
  _shape = shape;
  _output = output;
}

void BroadcastToLayer::run()
{
  // NOTE : It was implemented follows tf.broadcast_to operation works and
  //        Api Document(https://www.tensorflow.org/api_docs/python/tf/broadcast_to)

  switch (_output->data_type())
  {
    // ToDo : It need to support INT8 and UINT8 also when will be applied quantization.
    case OperandType::FLOAT32:
      nnfw::cker::BroadcastTo<float>(getShape(_input), reinterpret_cast<float *>(_input->buffer()),
                                     getShape(_output), getBuffer<float>(_output));
      break;
    case OperandType::INT32:
      nnfw::cker::BroadcastTo<int32_t>(getShape(_input),
                                       reinterpret_cast<int32_t *>(_input->buffer()),
                                       getShape(_output), getBuffer<int32_t>(_output));
      break;
    case OperandType::UINT32:
      nnfw::cker::BroadcastTo<uint32_t>(getShape(_input),
                                        reinterpret_cast<uint32_t *>(_input->buffer()),
                                        getShape(_output), getBuffer<uint32_t>(_output));
      break;
    default:
      throw std::runtime_error{"BroadcastToLayer: unsupported data type"};
  }
}

} // namespace ops
} // namespace training
} // namespace backend
} // namespace onert
