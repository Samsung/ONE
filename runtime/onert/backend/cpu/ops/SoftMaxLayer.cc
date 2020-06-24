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
namespace cpu
{
namespace ops
{

SoftMaxLayer::SoftMaxLayer() : _input(nullptr), _output(nullptr), _beta(0.0)
{
  // DO NOTHING
}

// Performs softmax along the input of size (input_size * batch_size).
void Softmax(const float *in, const int input_size, const int batch_size, const float beta,
             float *out)
{
  assert(input_size > 0);

  // For each batch
  for (int b = 0; b < batch_size; b++)
  {
    // Find the max coeff.
    float max_coeff = in[0];
    for (int i = 1; i < input_size; i++)
    {
      if (in[i] > max_coeff)
        max_coeff = in[i];
    }

    // Compute the normalized sum of exps.
    float exp_sum = 0.0;
    for (int i = 0; i < input_size; i++)
    {
      out[i] = std::exp((in[i] - max_coeff) * beta);
      exp_sum += out[i];
    }

    // Divide by the sum of exps.
    float reciprocal_sum_exp = 1.f / exp_sum;
    for (int i = 0; i < input_size; i++)
    {
      out[i] *= reciprocal_sum_exp;
    }

    // Advance in and out pointers for the next batch.
    in += input_size;
    out += input_size;
  }
}

void SoftMaxLayer::softmaxFloat32()
{
  if (getNumberOfDimensions(_input) == 2)
  {
    uint32_t batch_size = getSizeOfDimension(_input, 0);
    if (batch_size == 0)
      throw std::runtime_error("batch_size should not be 0");

    uint32_t input_size = getNumberOfElements(_input) / batch_size;
    Softmax(reinterpret_cast<const float *>(_input->buffer()), input_size, batch_size, _beta,
            reinterpret_cast<float *>(_output->buffer()));
  }
  else if (getNumberOfDimensions(_input) == 4)
  {
    nnfw::cker::SoftmaxParams op_params;
    op_params.beta = _beta;
    nnfw::cker::Softmax(op_params, getTensorShape(_input),
                        reinterpret_cast<const float *>(_input->buffer()), getTensorShape(_output),
                        reinterpret_cast<float *>(_output->buffer()));
  }
  else
  {
    throw std::runtime_error{"only 2D and 4D tensors supported"};
  }
}

void SoftMaxLayer::softmaxQuant8()
{
  nnfw::cker::Shape descrIn4D(4);

  if (getNumberOfDimensions(_input) == 2)
  {
    auto batch_size = getSizeOfDimension(_input, 0);
    if (batch_size == 0)
      throw std::runtime_error("batch_size should not be 0");

    auto input_size = getNumberOfElements(_input) / batch_size;
    descrIn4D.SetDim(0, batch_size);
    descrIn4D.SetDim(1, 1);
    descrIn4D.SetDim(2, 1);
    descrIn4D.SetDim(3, input_size);
  }
  else if (getNumberOfDimensions(_input) == 4)
  {
    descrIn4D.SetDim(0, _input->dimension(0));
    descrIn4D.SetDim(1, _input->dimension(1));
    descrIn4D.SetDim(2, _input->dimension(2));
    descrIn4D.SetDim(3, _input->dimension(3));
  }
  else
  {
    throw std::runtime_error{"only 2D and 4D tensors supported"};
  }
  if (_output->data_offset() != 0 || _output->data_scale() != 1.f / 256)
  {
    throw std::runtime_error{"incorrect scale / offset for output"};
  }
  static const int32_t kScaledDiffIntegerBits = 5;
  const double input_beta_real_multiplier = std::min(
      1.0 * _beta * _input->data_scale() * (1 << (31 - kScaledDiffIntegerBits)), (1ll << 31) - 1.0);
  int32_t input_multiplier = 0;
  int32_t input_left_shift = 0;
  QuantizeMultiplierGreaterThanOne(input_beta_real_multiplier, &input_multiplier,
                                   &input_left_shift);
  float diff_min = -1.0f * CalculateInputRadius(kScaledDiffIntegerBits, input_left_shift);

  nnfw::cker::SoftmaxParams op_params;
  op_params.input_multiplier = input_multiplier;
  op_params.input_left_shift = input_left_shift;
  op_params.diff_min = diff_min;
  nnfw::cker::Softmax(op_params, descrIn4D, reinterpret_cast<const uint8_t *>(_input->buffer()),
                      descrIn4D, reinterpret_cast<uint8_t *>(_output->buffer()));
}

void SoftMaxLayer::configure(const IPortableTensor *input, const float beta,
                             IPortableTensor *output)
{
  _input = input;
  _output = output;
  _beta = beta;
}

void SoftMaxLayer::run()
{
  if (_input->data_type() == OperandType::FLOAT32)
  {
    softmaxFloat32();
  }
  else if (_input->data_type() == OperandType::QUANT_UINT8_ASYMM)
  {
    softmaxQuant8();
  }
  else
  {
    throw std::runtime_error{"SoftMax: unsupported data type"};
  }
}

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert
