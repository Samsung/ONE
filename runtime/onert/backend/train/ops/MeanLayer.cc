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

#include "MeanLayer.h"

#include "OperationUtils.h"

#include <cker/Shape.h>
#include <cker/train/operation/ReduceMean.h>

namespace onert
{
namespace backend
{
namespace train
{
namespace ops
{

MeanLayer::MeanLayer()
  : cpu::ops::MeanLayer(), _back_prop_input{nullptr}, _back_prop_output{nullptr}
{
  // DO NOTHING
}

void MeanLayer::configure(const IPortableTensor *input, const IPortableTensor *axes, IPortableTensor *output,
                          bool keep_dims, IPortableTensor *back_prop_input, const IPortableTensor *back_prop_output)
{
  cpu::ops::MeanLayer::configure(input, axes, output, keep_dims);

  _back_prop_input = back_prop_input;
  _back_prop_output = back_prop_output;
}

void MeanLayer::forward(bool) { cpu::ops::MeanLayer::run(); }

void MeanLayer::backward()
{
  nnfw::cker::Shape temp_shape;
  if (_keep_dims == false)
  {
    temp_shape.ReplaceWith(getShape(_input));
    assert(getShape(_axes).DimensionsCount() == 1);
    for (int i = 0; i < getShape(_axes).Dims(0); ++i)
    {
      temp_shape.SetDim(reinterpret_cast<int32_t*>(_axes->buffer())[i], 1);
    }
  }
  else
  {
    temp_shape.ReplaceWith(getShape(_back_prop_input));
  }

  assert(_back_prop_output->data_type() == _input->data_type());
  switch (_back_prop_output->data_type())
  {
    case OperandType::FLOAT32:
    {
      nnfw::cker::train::MeanGrad(temp_shape, getBuffer<float>(_back_prop_output),
                                  getShape(_back_prop_input), getBuffer<float>(_back_prop_input));
      break;
    }
    default:
      throw std::runtime_error("train MeanLayer: unsupported data type");
  }
}

} // namespace ops
} // namespace train
} // namespace backend
} // namespace onert
