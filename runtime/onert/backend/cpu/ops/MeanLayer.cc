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
namespace cpu
{
namespace ops
{

MeanLayer::MeanLayer() : _input(nullptr), _output(nullptr), _axes(), _keep_dims(false)
{
  // DO NOTHING
}

void MeanLayer::MeanFloat32()
{
  nnfw::cker::Mean(getTensorShape(_input), reinterpret_cast<const float *>(_input->buffer()),
                   getTensorShape(_output), reinterpret_cast<float *>(_output->buffer()), _axes);
}

void MeanLayer::MeanQuant8()
{
  nnfw::cker::MeanQ8Asymm(getTensorShape(_input),
                          reinterpret_cast<const uint8_t *>(_input->buffer()), _input->data_scale(),
                          _input->data_offset(), getTensorShape(_output),
                          reinterpret_cast<uint8_t *>(_output->buffer()), _output->data_scale(),
                          _output->data_offset(), _axes);
}

void MeanLayer::configure(const IPortableTensor *input, IPortableTensor *output,
                          const std::vector<int> &axes, bool keep_dims)
{
  _input = input;
  _output = output;
  _axes = axes;
  _keep_dims = keep_dims;
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
} // namespace cpu
} // namespace backend
} // namespace onert
