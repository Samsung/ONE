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

#include "SplitLayer.h"

#include "OperationUtils.h"

#include <cker/operation/Split.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
{

SplitLayer::SplitLayer() : _input(nullptr), _num_splits(0), _axis(0), _outputs()
{
  // DO NOTHING
}

template <typename T> void SplitLayer::splitGeneric(void)
{
  nnfw::cker::SplitParams op_params;
  op_params.axis = _axis;
  op_params.num_split = _num_splits;

  std::vector<T *> outputPtrs;

  for (const auto output : _outputs)
  {
    outputPtrs.emplace_back(reinterpret_cast<T *>(output->buffer()));
  }

  nnfw::cker::Split<T>(op_params, getTensorShape(_input), reinterpret_cast<T *>(_input->buffer()),
                       getTensorShape(_outputs[0]), outputPtrs.data());
}

void SplitLayer::configure(const Tensor *input, uint16_t num_splits, int16_t axis,
                           std::vector<Tensor *> &outputs)
{
  assert(input != nullptr);

  _num_splits = num_splits;
  _input = input;
  _axis = axis;
  _outputs = outputs;
}

void SplitLayer::run()
{
  if (_input->data_type() == OperandType::FLOAT32)
  {
    splitGeneric<float>();
  }
  else if (_input->data_type() == OperandType::QUANT_UINT8_ASYMM)
  {
    splitGeneric<uint8_t>();
  }
  else
  {
    throw std::runtime_error{"Split: unsupported input type"};
  }
}

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert
