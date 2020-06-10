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

#include "MatrixBandPartLayer.h"

#include "OperationUtils.h"

#include <cker/operation/MatrixBandPart.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
{

MatrixBandPartLayer::MatrixBandPartLayer()
    : _input(nullptr), _num_lower_diag(nullptr), _num_upper_diag(nullptr), _output(nullptr)
{
  // DO NOTHING
}

void MatrixBandPartLayer::matrixBandPartFloat32()
{
  auto num_lower_diag = *reinterpret_cast<const int *>(_num_lower_diag->buffer());
  auto num_upper_diag = *reinterpret_cast<const int *>(_num_upper_diag->buffer());

  nnfw::cker::MatrixBandPart(num_lower_diag, num_upper_diag, getTensorShape(_input),
                             reinterpret_cast<const float *>(_input->buffer()),
                             getTensorShape(_output), reinterpret_cast<float *>(_output->buffer()));
}

void MatrixBandPartLayer::matrixBandPartQuant8() { throw std::runtime_error{"NYI"}; }

void MatrixBandPartLayer::configure(const Tensor *input, const Tensor *num_lower_diag,
                                    const Tensor *num_upper_diag, Tensor *output)
{
  _input = input;
  _num_lower_diag = num_lower_diag;
  _num_upper_diag = num_upper_diag;
  _output = output;
}

void MatrixBandPartLayer::run()
{
  if (_input->data_type() == OperandType::FLOAT32)
  {
    matrixBandPartFloat32();
  }
  else if (_input->data_type() == OperandType::QUANT_UINT8_ASYMM)
  {
    matrixBandPartQuant8();
  }
  else
  {
    throw std::runtime_error{"MatrixBandpart: unsupported data type"};
  }
}

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert
