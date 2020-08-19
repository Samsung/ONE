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

#include "PoolLayer.h"

#include <cker/operation/AveragePool.h>
#include <cker/operation/MaxPool.h>

#include <unordered_map>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
{

namespace
{
template <typename T>
void pool2D(const nnfw::cker::PoolParams &params, const nnfw::cker::Shape &input_shape,
            const T *input_data, const nnfw::cker::Shape &output_shape, T *output_data,
            PoolType op_type)
{
  if (op_type == PoolType::kAvg)
  {
    nnfw::cker::AveragePool<T>(params, input_shape, input_data, output_shape, output_data);
  }
  else if (op_type == PoolType::kMax)
  {
    nnfw::cker::MaxPool<T>(params, input_shape, input_data, output_shape, output_data);
  }
  else
  {
    throw std::runtime_error{"Pool: unsupported pool type"};
  }
}
} // namespace

PoolLayer::PoolLayer()
    : _input(nullptr), _output(nullptr), _paddingLeft(0), _paddingTop(0), _paddingRight(0),
      _paddingBottom(0), _strideWidth(0), _strideHeight(0), _kernelWidth(0), _kernelHeight(0),
      _activation(ir::Activation::NONE), _op_type(PoolType::kAvg)
{
  // DO NOTHING
}

#define POOLING_PARAMETERS                               \
  nnfw::cker::PoolParams op_params;                      \
  op_params.stride_height = _strideHeight;               \
  op_params.stride_width = _strideWidth;                 \
  op_params.filter_height = _kernelHeight;               \
  op_params.filter_width = _kernelWidth;                 \
  op_params.padding_values.height = (int8_t)_paddingTop; \
  op_params.padding_values.width = (int8_t)_paddingLeft;

void PoolLayer::configure(const IPortableTensor *input, const uint32_t paddingLeft,
                          const uint32_t paddingRight, const uint32_t paddingTop,
                          const uint32_t paddingBottom, const uint32_t strideWidth,
                          const uint32_t strideHeight, const uint32_t kernelWidth,
                          const uint32_t kernelHeight, const ir::Activation activation,
                          IPortableTensor *output, const PoolType op_type)
{
  assert(input != nullptr);
  assert(output != nullptr);

  _input = input;
  _paddingLeft = paddingLeft;
  _paddingRight = paddingRight;
  _paddingTop = paddingTop;
  _paddingBottom = paddingBottom;
  _strideWidth = strideWidth;
  _strideHeight = strideHeight;
  _kernelWidth = kernelWidth;
  _kernelHeight = kernelHeight;
  _activation = activation;
  _op_type = op_type;
  _output = output;

  if (op_type != PoolType::kAvg && op_type != PoolType::kMax)
  {
    throw std::runtime_error{"Pool: unsupported pool type"};
  }
}

void PoolLayer::run()
{
  if (_input->data_type() == OperandType::FLOAT32)
  {
    POOLING_PARAMETERS
    float output_activation_min = 0;
    float output_activation_max = 0;
    CalculateActivationRange<float>(_activation, &output_activation_min, &output_activation_max);
    op_params.float_activation_min = output_activation_min;
    op_params.float_activation_max = output_activation_max;

    pool2D<float>(op_params, getTensorShape(_input),
                  reinterpret_cast<const float *>(_input->buffer()), getTensorShape(_output),
                  reinterpret_cast<float *>(_output->buffer()), _op_type);
  }
  else if (_input->data_type() == OperandType::QUANT_UINT8_ASYMM)
  {
    POOLING_PARAMETERS
    int32_t output_activation_min = 0;
    int32_t output_activation_max = 0;
    CalculateActivationRangeUint8(_activation, _output, &output_activation_min,
                                  &output_activation_max);
    op_params.quantized_activation_min = output_activation_min;
    op_params.quantized_activation_max = output_activation_max;
    pool2D<uint8_t>(op_params, getTensorShape(_input),
                    reinterpret_cast<const uint8_t *>(_input->buffer()), getTensorShape(_output),
                    reinterpret_cast<uint8_t *>(_output->buffer()), _op_type);
  }
  else
  {
    throw std::runtime_error{"Pool: unsupported data type"};
  }
}

#undef AVGPOOLING_PARAMETERS

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert
