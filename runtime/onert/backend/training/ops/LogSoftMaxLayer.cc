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

#include "LogSoftMaxLayer.h"

#include "OperationUtils.h"

#include <cker/operation/LogSoftMax.h>

namespace onert
{
namespace backend
{
namespace training
{
namespace ops
{

LogSoftMaxLayer::LogSoftMaxLayer() : _input(nullptr), _output(nullptr), _beta(0.0), _axis(0)
{
  // DO NOTHING
}

void LogSoftMaxLayer::PopulateLookupTable(const float kBeta)
{
  const float scale = -_input->data_scale() * kBeta;
  const int32_t max_uint8 = std::numeric_limits<uint8_t>::max();
  for (int32_t val = 0; val <= max_uint8; ++val)
  {
    _table[max_uint8 - val] = expf(scale * val);
  }
}

void LogSoftMaxLayer::logsoftmaxFloat32()
{
  nnfw::cker::SoftmaxParams op_params;
  op_params.beta = _beta;
  op_params.axis = _axis;
  nnfw::cker::LogSoftmax(op_params, getShape(_input), getBuffer<float>(_input), getShape(_output),
                         getBuffer<float>(_output));
}

void LogSoftMaxLayer::logsoftmaxQuant8()
{
  nnfw::cker::SoftmaxParams op_params;
  op_params.beta = _beta;
  op_params.axis = _axis;
  op_params.table = _table;
  op_params.zero_point = _output->data_zero_point();
  op_params.scale = _output->data_scale();
  nnfw::cker::LogSoftmax(op_params, _input->data_scale(), getShape(_input),
                         getBuffer<uint8_t>(_input), getShape(_output),
                         getBuffer<uint8_t>(_output));
}

void LogSoftMaxLayer::configure(const IPortableTensor *input, const float beta, const int axis,
                                IPortableTensor *output)
{
  _input = input;
  _output = output;
  _beta = beta;
  _axis = axis;
  if (_input->data_type() == OperandType::QUANT_UINT8_ASYMM)
  {
    PopulateLookupTable(_beta);
  }
}

void LogSoftMaxLayer::run()
{
  if (_input->data_type() == OperandType::FLOAT32)
  {
    logsoftmaxFloat32();
  }
  else if (_input->data_type() == OperandType::QUANT_UINT8_ASYMM)
  {
    logsoftmaxQuant8();
  }
  else
  {
    throw std::runtime_error{"LogSoftmax : unsupported data type"};
  }
}

} // namespace ops
} // namespace training
} // namespace backend
} // namespace onert
