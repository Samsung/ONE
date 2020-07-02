/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ReshapeLayer.h"

namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
{

ReshapeLayer::ReshapeLayer() : _input(nullptr), _shape(nullptr), _output(nullptr)
{
  // DO NOTHING
}

void ReshapeLayer::reshapeGeneric()
{
  size_t count = _input->total_size();
  memcpy(_output->buffer(), _input->buffer(), count);
}

void ReshapeLayer::configure(const IPortableTensor *input, const IPortableTensor *shape,
                             IPortableTensor *output)
{
  _input = input;
  /* note : shape is optional. If not provided from model, _shape is nullptr. */
  _shape = shape;
  _output = output;
}

void ReshapeLayer::run() { reshapeGeneric(); }

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert
