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
namespace cpu
{
namespace ops
{

LogSoftMaxLayer::LogSoftMaxLayer() : _input(nullptr), _output(nullptr), _beta(0.0), _axis(0)
{
  // DO NOTHING
}

void LogSoftMaxLayer::logsoftmaxFloat32()
{
  nnfw::cker::SoftmaxParams op_params;
  op_params.beta = _beta;
  op_params.axis = _axis;
  nnfw::cker::LogSoftmax(op_params, getTensorShape(_input),
                         reinterpret_cast<const float *>(_input->buffer()), getTensorShape(_output),
                         reinterpret_cast<float *>(_output->buffer()));
}

void LogSoftMaxLayer::logsoftmaxQuant8()
{
  // NYI
}

void LogSoftMaxLayer::configure(const IPortableTensor *input, const float beta, const int axis,
                                IPortableTensor *output)
{
  _input = input;
  _output = output;
  _beta = beta;
  _axis = axis;
}

void LogSoftMaxLayer::run()
{
  if (_input->data_type() == OperandType::FLOAT32)
  {
    logsoftmaxFloat32();
  }
  else if (_input->data_type() == OperandType::QUANT_UINT8_ASYMM)
  {
    throw std::runtime_error{"LogSoftmax : NYI"};
  }
  else
  {
    throw std::runtime_error{"LogSoftmax : unsupported data type"};
  }
}

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert
