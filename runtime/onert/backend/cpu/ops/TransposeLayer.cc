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

#include "TransposeLayer.h"

#include "OperationUtils.h"

#include <cker/operation/Transpose.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
{

TransposeLayer::TransposeLayer() : _input(nullptr), _output(nullptr), _perm()
{
  // DO NOTHING
}

void TransposeLayer::transposeFloat32()
{
  nnfw::cker::TransposeParams param;
  param.perm_count = _perm.size();
  for (size_t i = 0; i < _perm.size(); i++)
  {
    param.perm[i] = _perm[i];
  }

  nnfw::cker::Transpose(param, getTensorShape(_input),
                        reinterpret_cast<const float *>(_input->buffer()), getTensorShape(_output),
                        reinterpret_cast<float *>(_output->buffer()));
}

void TransposeLayer::transposeQuant8()
{
  // cker quant8 tanh is not implemented yet
  throw std::runtime_error{"NYI"};
}

void TransposeLayer::configure(const IPortableTensor *input, IPortableTensor *output,
                               const std::vector<int> &perm)
{
  _input = input;
  _perm = perm;
  _output = output;
}

void TransposeLayer::run()
{
  if (_input->data_type() == OperandType::FLOAT32)
  {
    transposeFloat32();
  }
  else if (_input->data_type() == OperandType::QUANT_UINT8_ASYMM)
  {
    transposeQuant8();
  }
  else
  {
    throw std::runtime_error{"Transpose: unsupported data type"};
  }
}

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert
