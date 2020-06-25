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

#include "PackLayer.h"

#include "OperationUtils.h"

#include <cker/operation/Pack.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
{

PackLayer::PackLayer() : _inputs(), _output(nullptr), _axis(0)
{
  // DO NOTHING
}

template <typename T> void PackLayer::packImpl()
{
  uint32_t num_inputs = _inputs.size();
  nnfw::cker::PackParams op_params;
  op_params.axis = _axis;
  op_params.inputs_count = num_inputs;

  std::vector<nnfw::cker::Shape *> inputDimsPtr;
  std::vector<nnfw::cker::Shape> inputDims;
  inputDimsPtr.reserve(num_inputs);
  inputDims.reserve(num_inputs);

  for (uint32_t i = 0; i < num_inputs; i++)
  {
    inputDims.push_back(getTensorShape(_inputs[i]));
    inputDimsPtr.push_back(&inputDims[i]);
  }

  std::vector<const T *> inputPtrs;

  for (const auto input : _inputs)
  {
    inputPtrs.emplace_back(reinterpret_cast<const T *>(input->buffer()));
  }

  nnfw::cker::Pack<T>(op_params, inputPtrs.data(), getTensorShape(_output),
                      reinterpret_cast<T *>(_output->buffer()));
}

void PackLayer::configure(const std::vector<const IPortableTensor *> &inputs, int32_t axis,
                          IPortableTensor *output)
{
  assert(inputs.size() > 0);
  assert(output != nullptr);

  _inputs = inputs;
  _axis = axis;
  _output = output;
}

void PackLayer::run()
{
  if (_output->data_type() == OperandType::FLOAT32)
  {
    packImpl<float>();
  }
  else if (_output->data_type() == OperandType::INT32)
  {
    packImpl<int32_t>();
  }
  else
  {
    throw std::runtime_error{"Pack: unsupported data type"};
  }
}

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert
