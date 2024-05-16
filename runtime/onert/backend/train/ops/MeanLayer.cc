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
#include <cker/operation/BinaryArithmeticOps.h>

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

void MeanLayer::configureBackward(IPortableTensor *back_prop_input,
                                  const IPortableTensor *back_prop_output)
{
  _back_prop_input = back_prop_input;
  _back_prop_output = back_prop_output;
}

void MeanLayer::forward(bool) { cpu::ops::MeanLayer::run(); }

void MeanLayer::backward()
{
  nnfw::cker::Shape keep_dim_shape;
  if (_keep_dims == false)
  {
    keep_dim_shape.ReplaceWith(getShape(_input));
    auto axes_vec = cpu::ops::getReducerAxes(_axes);
    for (const auto &axis : axes_vec)
    {
      keep_dim_shape.SetDim(axis, 1);
    }
  }
  else
  {
    keep_dim_shape.ReplaceWith(getShape(_back_prop_input));
  }

  switch (_back_prop_output->data_type())
  {
    case OperandType::FLOAT32:
    {
      nnfw::cker::train::MeanGrad(keep_dim_shape, getBuffer<float>(_back_prop_output),
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
