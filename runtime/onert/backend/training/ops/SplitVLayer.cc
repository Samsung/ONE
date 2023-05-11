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

#include "SplitVLayer.h"

#include "OperationUtils.h"

#include <cker/operation/SplitV.h>

namespace onert
{
namespace backend
{
namespace training
{
namespace ops
{

SplitVLayer::SplitVLayer()
  : _input(nullptr), _size_splits(nullptr), _split_dim(nullptr), _num_splits(0), _outputs()
{
  // DO NOTHING
}

template <typename T> void SplitVLayer::splitV(void)
{
  nnfw::cker::SplitVParams op_params;
  op_params.axis = *getBuffer<int32_t>(_split_dim);
  op_params.num_split = _num_splits;

  std::vector<T *> outputPtrs;
  std::vector<nnfw::cker::Shape> outshape;

  for (const auto output : _outputs)
  {
    assert(output->total_size() == sizeOfData(output->data_type(), output->getShape().dims()));
    outputPtrs.emplace_back(getBuffer<T>(output));
    outshape.emplace_back(getShape(output));
  }

  assert(_input->total_size() == sizeOfData(_input->data_type(), _input->getShape().dims()));
  nnfw::cker::SplitV<T>(op_params, getShape(_input), getBuffer<T>(_input), outshape,
                        outputPtrs.data());
}

void SplitVLayer::configure(const IPortableTensor *input, const IPortableTensor *size_splits,
                            const IPortableTensor *split_dim, uint16_t num_splits,
                            std::vector<IPortableTensor *> &outputs)
{
  assert(input != nullptr);

  _num_splits = num_splits;
  _size_splits = size_splits;
  _input = input;
  _split_dim = split_dim;
  _outputs = outputs;
}

void SplitVLayer::run()
{
  if (_input->data_type() == OperandType::FLOAT32)
  {
    splitV<float>();
  }
  else if (_input->data_type() == OperandType::QUANT_UINT8_ASYMM)
  {
    splitV<uint8_t>();
  }
  else if (_input->data_type() == OperandType::INT32)
  {
    splitV<int32_t>();
  }
  else if (_input->data_type() == OperandType::INT64)
  {
    splitV<int64_t>();
  }
  else
  {
    throw std::runtime_error{"SplitV: unsupported input type"};
  }
}

} // namespace ops
} // namespace training
} // namespace backend
} // namespace onert
