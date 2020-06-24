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

void SplitLayer::splitFloat32()
{
  nnfw::cker::SplitParams op_params;
  op_params.axis = _axis;
  op_params.num_split = _num_splits;

  std::vector<nnfw::cker::Shape *> outputDimsPtr;
  std::vector<nnfw::cker::Shape> outputDims;
  outputDimsPtr.reserve(_num_splits);
  outputDims.reserve(_num_splits);

  for (uint32_t i = 0; i < _num_splits; i++)
  {
    outputDims.push_back(getTensorShape(_outputs[i]));
    outputDimsPtr.push_back(&outputDims[i]);
  }

  std::vector<float *> outputFloatPtrs;

  for (const auto output : _outputs)
  {
    outputFloatPtrs.emplace_back(reinterpret_cast<float *>(output->buffer()));
  }

  nnfw::cker::Split<float>(op_params, getTensorShape(_input),
                           reinterpret_cast<float *>(_input->buffer()), getTensorShape(_outputs[0]),
                           outputFloatPtrs.data());
}

void SplitLayer::splitQuant8() { throw std::runtime_error{"Split: NYI quant8 type"}; }

void SplitLayer::configure(const IPortableTensor *input, uint16_t num_splits, int16_t axis,
                           std::vector<IPortableTensor *> &outputs)
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
    splitFloat32();
  }
  else if (_input->data_type() == OperandType::QUANT_UINT8_ASYMM)
  {
    splitQuant8();
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
