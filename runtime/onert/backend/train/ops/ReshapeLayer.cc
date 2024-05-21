/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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
namespace train
{
namespace ops
{

ReshapeLayer::ReshapeLayer()
  : _input{nullptr}, _shape{nullptr}, _output{nullptr}, _back_prop_input{nullptr},
    _back_prop_output{nullptr}
{
  // DO NOTHING
}

void ReshapeLayer::reshapeGeneric(const IPortableTensor *input, IPortableTensor *output)
{
  size_t count = input->total_size();
  memcpy(output->buffer(), input->buffer(), count);
}

void ReshapeLayer::configure(const IPortableTensor *input, const IPortableTensor *shape,
                             IPortableTensor *output)
{
  _input = input;
  /* note : shape is optional. If not provided from model, _shape is nullptr. */
  _shape = shape;
  _output = output;
}

void ReshapeLayer::configureBackward(IPortableTensor *back_prop_input,
                                     const IPortableTensor *back_prop_output)
{
  _back_prop_input = back_prop_input;
  _back_prop_output = back_prop_output;
}

void ReshapeLayer::forward(bool) { reshapeGeneric(_input, _output); }

void ReshapeLayer::backward() { reshapeGeneric(_back_prop_output, _back_prop_input); }

} // namespace ops
} // namespace train
} // namespace backend
} // namespace onert
