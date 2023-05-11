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

#include "SoftMaxLayer.h"

#include "OperationUtils.h"

#include <cker/operation/SoftMax.h>

namespace onert
{
namespace backend
{
namespace training
{
namespace ops
{

SoftMaxLayer::SoftMaxLayer() : _input(nullptr), _output(nullptr), _beta(0.0)
{
  // DO NOTHING
}

void SoftMaxLayer::softmaxFloat32()
{
  if (getNumberOfDimensions(_input) == 1)
  {
    uint32_t input_size = getNumberOfElements(_input);
    nnfw::cker::Softmax(getBuffer<float>(_input), input_size, 1, _beta, getBuffer<float>(_output));
  }
  else if (getNumberOfDimensions(_input) == 2)
  {
    uint32_t batch_size = getSizeOfDimension(_input, 0);
    if (batch_size == 0)
      throw std::runtime_error("batch_size should not be 0");

    uint32_t input_size = getNumberOfElements(_input) / batch_size;
    nnfw::cker::Softmax(getBuffer<float>(_input), input_size, batch_size, _beta,
                        getBuffer<float>(_output));
  }
  else if (getNumberOfDimensions(_input) == 4)
  {
    nnfw::cker::SoftmaxParams op_params;
    op_params.beta = _beta;
    nnfw::cker::Softmax(op_params, getShape(_input), getBuffer<float>(_input), getShape(_output),
                        getBuffer<float>(_output));
  }
  else
  {
    nnfw::cker::SoftmaxParams op_params;
    op_params.beta = _beta;
    nnfw::cker::reference::Softmax(op_params, getShape(_input), getBuffer<float>(_input),
                                   getShape(_output), getBuffer<float>(_output));
  }
}

template <typename T> void SoftMaxLayer::softmaxQuant8()
{
  nnfw::cker::SoftmaxParams op_params;
  op_params.scale = _output->data_scale();
  op_params.zero_point = _output->data_zero_point();
  op_params.uint8_table1 = _uint8_table1;
  op_params.uint8_table2 = _uint8_table2;
  op_params.table = _table;

#ifdef TFLITE_SOFTMAX_USE_UINT16_LUT
  nnfw::cker::SoftmaxInt8LUT<T, T>(op_params, getShape(_input), getBuffer<T>(_input),
                                   getShape(_output), getBuffer<T>(_output));
#else
  nnfw::cker::Softmax<T, T>(op_params, getShape(_input), getBuffer<T>(_input), getShape(_output),
                            getBuffer<T>(_output));
#endif
}

void SoftMaxLayer::configure(const IPortableTensor *input, const float beta,
                             IPortableTensor *output)
{
  _input = input;
  _output = output;
  _beta = beta;

  if (_input->data_type() == OperandType::QUANT_UINT8_ASYMM ||
      _input->data_type() == OperandType::QUANT_INT8_ASYMM)
  {
#ifdef TFLITE_SOFTMAX_USE_UINT16_LUT
    // Only apply when both input & output are uint8/int8 & build with clang
    // on aarch64.
    nnfw::cker::PopulateSoftmaxUInt8LookupTable(_uint8_table1, _uint8_table2, _input->data_scale(),
                                                _beta);
#else
    nnfw::cker::PopulateSoftmaxLookupTable(_table, _input->data_scale(), _beta);
#endif
  }
}

void SoftMaxLayer::run()
{
  switch (_input->data_type())
  {
    case OperandType::FLOAT32:
      softmaxFloat32();
      break;
    case OperandType::QUANT_UINT8_ASYMM:
      softmaxQuant8<uint8_t>();
      break;
    case OperandType::QUANT_INT8_ASYMM:
      softmaxQuant8<int8_t>();
      break;
    default:
      throw std::runtime_error{"SoftMax: unsupported data type"};
  }
}

} // namespace ops
} // namespace training
} // namespace backend
} // namespace onert
