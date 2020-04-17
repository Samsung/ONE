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

#include <util/ShapeInference.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace kernel
{

ReshapeLayer::ReshapeLayer() : _input(nullptr), _shape(nullptr), _output(nullptr)
{
  // DO NOTHING
}

void ReshapeLayer::reshapeGeneric()
{
  // if dynamic allocation is needed, allocate output tensor
  if (_output->is_dynamic())
  {
    auto output_shape = shape_inference::dynamic_inf::inferReshape(_shape);
    _output->info().shape(output_shape);

    // set dynamic tensor's buffer
    assert(_output->buffer() == nullptr);

    //
    // TODO Write code for memory allocation
    //
    throw std::runtime_error("Not yet implemented.");
  }

  // perform resize operation
  size_t count = _input->total_size();
  memcpy(_output->buffer(), _input->buffer(), count);
}

void ReshapeLayer::configure(const operand::Tensor *input, const operand::Tensor *shape,
                             operand::Tensor *output)
{
  _input = input;
  /* note : shape is optional. If not provided from model, _shape is nullptr. */
  _shape = shape;
  _output = output;
}

void ReshapeLayer::run() { reshapeGeneric(); }

} // namespace kernel
} // namespace cpu
} // namespace backend
} // namespace onert
