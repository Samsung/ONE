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
namespace kernel
{

UnpackLayer::UnpackLayer() : _input(nullptr), _outputs(), _axis(0), _num_output(0)
{
  // DO NOTHING
}

void UnpackLayer::unpackFloat32()
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
    outputDims.push_back(convertTensorToCkerShape(_outputs[i]));
    outputDimsPtr.push_back(&outputDims[i]);
  }

  std::vector<float *> outputFloatPtrs;

  for (const auto output : _outputs)
  {
    outputFloatPtrs.emplace_back(reinterpret_cast<float *>(output->buffer()));
  }

  nnfw::cker::Unpack<float>(op_params, convertTensorToCkerShape(_input),
                            reinterpret_cast<float *>(_input->buffer()),
                            convertTensorToCkerShape(_outputs[0]), outputFloatPtrs.data());
}

void UnpackLayer::unpackQuant8()
{
  // cker quant8 pack is not implemented yet
  throw std::runtime_error{"Unpack: NYI quant8 type"};
}

void UnpackLayer::configure(const ITensor *input, uint32_t axis, int32_t num,
                            std::vector<ITensor *> &outputs)
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
  {
    unpackFloat32();
  }
  else if (_input->data_type() == OperandType::QUANT8_ASYMM)
  {
    unpackQuant8();
  }
  else
  {
    throw std::runtime_error{"Unpack: Unsupported input type"};
  }
}

} // namespace kernel
} // namespace cpu
} // namespace backend
} // namespace onert
