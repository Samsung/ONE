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

#include "UnpackLayer.h"

#include "OperationUtils.h"

#include <cker/operation/Unpack.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
{

UnpackLayer::UnpackLayer() : _input(nullptr), _outputs(), _axis(0), _num_output(0)
{
  // DO NOTHING
}

template <typename T> void UnpackLayer::unpackImpl()
{
  nnfw::cker::UnpackParams op_params;
  op_params.axis = _axis;
  op_params.num_split = _num_output;

  std::vector<nnfw::cker::Shape *> outputDimsPtr;
  std::vector<nnfw::cker::Shape> outputDims;
  outputDimsPtr.reserve(_num_output);
  outputDims.reserve(_num_output);

  for (int32_t i = 0; i < _num_output; i++)
  {
    outputDims.push_back(getTensorShape(_outputs[i]));
    outputDimsPtr.push_back(&outputDims[i]);
  }

  std::vector<T *> outputPtrs;

  for (const auto output : _outputs)
  {
    outputPtrs.emplace_back(reinterpret_cast<T *>(output->buffer()));
  }

  nnfw::cker::Unpack<T>(op_params, getTensorShape(_input), reinterpret_cast<T *>(_input->buffer()),
                        getTensorShape(_outputs[0]), outputPtrs.data());
}

void UnpackLayer::configure(const IPortableTensor *input, uint32_t axis, int32_t num,
                            std::vector<IPortableTensor *> &outputs)
{
  assert(input != nullptr);
  assert(outputs.size() > 0);
  assert(outputs.size() == (size_t)num);

  _input = input;
  _axis = axis;
  _num_output = num;
  _outputs = outputs;
}

void UnpackLayer::run()
{
  if (_input->data_type() == OperandType::FLOAT32)
    unpackImpl<float>();
  else if (_input->data_type() == OperandType::INT32)
    unpackImpl<int32_t>();
  else
    throw std::runtime_error{"Unpack: Unsupported data type"};
}

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert
