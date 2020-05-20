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
namespace kernel
{

PackLayer::PackLayer() : _inputs(), _output(nullptr), _axis(0)
{
  // DO NOTHING
}

void PackLayer::packFloat32()
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
    inputDims.push_back(convertTensorToCkerShape(_inputs[i]));
    inputDimsPtr.push_back(&inputDims[i]);
  }

  std::vector<const float *> inputFloatPtrs;

  for (const auto input : _inputs)
  {
    inputFloatPtrs.emplace_back(reinterpret_cast<const float *>(input->buffer()));
  }

  nnfw::cker::Pack<float>(op_params, inputFloatPtrs.data(), convertTensorToCkerShape(_output),
                          reinterpret_cast<float *>(_output->buffer()));
}

void PackLayer::packQuant8()
{
  // cker quant8 pack is not implemented yet
  throw std::runtime_error{"NYI"};
}

void PackLayer::configure(const std::vector<const ITensor *> &inputs, int32_t axis,
                          ITensor *output)
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
    packFloat32();
  }
  else if (_output->data_type() == OperandType::QUANT8_ASYMM)
  {
    packQuant8();
  }
}

} // namespace kernel
} // namespace cpu
} // namespace backend
} // namespace onert
