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
#include <numeric>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
{

TransposeLayer::TransposeLayer() : _input(nullptr), _perm(nullptr), _output(nullptr)
{
  // DO NOTHING
}

template <typename T> void TransposeLayer::transpose()
{
  nnfw::cker::TransposeParams param;
  assert(_perm->num_dimensions() == 1);

  param.perm_count = _input->num_dimensions();
  if (_perm->dimension(0) == 0) // This means _perm is (n-1...0)
  {
    const auto begin = param.perm;
    const auto end = param.perm + _input->num_dimensions() - 1;
    std::iota(begin, end, 0);
    std::reverse(begin, end);
  }
  else
  {
    assert(param.perm_count == static_cast<int>(_perm->dimension(0)));
    for (auto i = 0; i < param.perm_count; i++)
    {
      param.perm[i] = *(reinterpret_cast<const int32_t *>(_perm->buffer()) + i);
    }
  }

  nnfw::cker::Transpose(param, getTensorShape(_input),
                        reinterpret_cast<const T *>(_input->buffer()), getTensorShape(_output),
                        reinterpret_cast<T *>(_output->buffer()));
}

void TransposeLayer::transposeQuant8()
{
  if (_input->data_offset() != _output->data_offset())
  {
    throw std::runtime_error("TransposeLayer : qassym8 input and output offsets unmatched");
  }

  if (_input->data_scale() != _output->data_scale())
  {
    throw std::runtime_error("TransposeLayer : qassym8 input and output scales unmatched");
  }

  transpose<uint8_t>();
}

void TransposeLayer::configure(const IPortableTensor *input, const IPortableTensor *perm,
                               IPortableTensor *output)
{
  _input = input;
  _perm = perm;
  _output = output;
}

void TransposeLayer::run()
{
  if (_input->data_type() == OperandType::FLOAT32)
  {
    transpose<float>();
  }
  else if (_input->data_type() == OperandType::INT32)
  {
    transpose<int32_t>();
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
