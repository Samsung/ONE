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

#include "MeanLayer.h"

#include "OperationUtils.h"

#include <cker/operation/ReduceMean.h>

namespace onert
{
namespace backend
{
namespace training
{
namespace ops
{

MeanLayer::MeanLayer() : _input(nullptr), _axes(nullptr), _output(nullptr), _keep_dims(false)
{
  // DO NOTHING
}

void MeanLayer::MeanFloat32()
{
  const auto inputShape = getShape(_input);
  const auto axisVec = getReducerAxes(_axes);
  bool axis_is_1_and_2 =
    _keep_dims && inputShape.DimensionsCount() == 4 && axisVec.size() == 2 &&
    ((axisVec[0] == 1 && axisVec[1] == 2) || (axisVec[0] == 2 && axisVec[1] == 1));

  if (axis_is_1_and_2)
  {
    nnfw::cker::MeanAxis1And2(inputShape, getBuffer<float>(_input), getShape(_output),
                              getBuffer<float>(_output));
  }
  else
  {
    nnfw::cker::Mean(inputShape, getBuffer<float>(_input), getShape(_output),
                     getBuffer<float>(_output), axisVec);
  }
}

void MeanLayer::MeanQuant8()
{
  nnfw::cker::MeanQ8Asymm(getShape(_input), getBuffer<uint8_t>(_input), _input->data_scale(),
                          _input->data_zero_point(), getShape(_output), getBuffer<uint8_t>(_output),
                          _output->data_scale(), _output->data_zero_point(), getReducerAxes(_axes));
}

void MeanLayer::configure(const IPortableTensor *input, const IPortableTensor *axes,
                          IPortableTensor *output, bool keep_dims)
{
  _input = input;
  _axes = axes;
  _output = output;
  _keep_dims = keep_dims;

  if (_input->data_type() != OperandType::FLOAT32 &&
      _input->data_type() != OperandType::QUANT_UINT8_ASYMM)
    throw std::runtime_error{"Mean: unsupported data type"};
}

void MeanLayer::run()
{
  if (_input->data_type() == OperandType::FLOAT32)
  {
    MeanFloat32();
  }
  else if (_input->data_type() == OperandType::QUANT_UINT8_ASYMM)
  {
    MeanQuant8();
  }
  else
  {
    throw std::runtime_error{"Mean: unsupported data type"};
  }
}

} // namespace ops
} // namespace training
} // namespace backend
} // namespace onert
