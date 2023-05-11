/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ConcatLayer.h"

#include "OperationUtils.h"

#include <cker/operation/Concatenation.h>

namespace onert
{
namespace backend
{
namespace training
{
namespace ops
{

ConcatLayer::ConcatLayer() : _inputs(), _output(nullptr), _axis(0)
{
  // DO NOTHING
}

template <typename T> void ConcatLayer::concatenationGeneral()
{
  uint32_t num_inputs = _inputs.size();

  nnfw::cker::ConcatenationParams op_params;
  op_params.axis = _axis;
  op_params.inputs_count = num_inputs;

  std::vector<nnfw::cker::Shape *> inputDimsPtr;
  std::vector<nnfw::cker::Shape> inputDims;
  inputDimsPtr.reserve(num_inputs);
  inputDims.reserve(num_inputs);

  for (uint32_t i = 0; i < num_inputs; i++)
  {
    inputDims.push_back(getShape(_inputs[i]));
    inputDimsPtr.push_back(&inputDims[i]);
  }

  std::vector<const T *> inputDataPtrs;

  for (const auto input : _inputs)
  {
    inputDataPtrs.emplace_back(getBuffer<T>(input));
  }

  nnfw::cker::Concatenation<T>(op_params, inputDimsPtr.data(), inputDataPtrs.data(),
                               getShape(_output), getBuffer<T>(_output));
}
void ConcatLayer::concatenationQuant8()
{
  uint32_t num_inputs = _inputs.size();

  std::vector<int32_t> input_zeropoints(num_inputs);
  std::vector<float> input_scales(num_inputs);
  for (uint32_t i = 0; i < num_inputs; i++)
  {
    input_zeropoints[i] = _inputs[i]->data_zero_point();
    input_scales[i] = _inputs[i]->data_scale();
  }

  nnfw::cker::ConcatenationParams op_params;
  op_params.axis = _axis;
  op_params.inputs_count = num_inputs;
  op_params.input_zeropoint = input_zeropoints.data();
  op_params.input_scale = input_scales.data();
  op_params.output_zeropoint = _output->data_zero_point();
  op_params.output_scale = _output->data_scale();

  std::vector<nnfw::cker::Shape *> inputDimsPtr;
  std::vector<nnfw::cker::Shape> inputDims;
  inputDimsPtr.reserve(num_inputs);
  inputDims.reserve(num_inputs);
  for (uint32_t i = 0; i < num_inputs; i++)
  {
    inputDims.push_back(getShape(_inputs[i]));
    inputDimsPtr.push_back(&inputDims[i]);
  }

  std::vector<const uint8_t *> inputDataPtrs;
  for (const auto input : _inputs)
  {
    inputDataPtrs.emplace_back(getBuffer<uint8_t>(input));
  }

  nnfw::cker::ConcatenationWithScaling(op_params, inputDimsPtr.data(), inputDataPtrs.data(),
                                       getShape(_output), getBuffer<uint8_t>(_output));
}

void ConcatLayer::configure(const std::vector<const IPortableTensor *> &inputs, int32_t axis,
                            IPortableTensor *output)
{
  assert(inputs.size() > 0);
  assert(output != nullptr);

  _inputs = inputs;
  _axis = axis;
  _output = output;
}

void ConcatLayer::run()
{
  switch (_output->data_type())
  {
    case OperandType::FLOAT32:
      concatenationGeneral<float>();
      break;
    case OperandType::QUANT_UINT8_ASYMM:
      concatenationQuant8();
      break;
    case OperandType::QUANT_INT8_ASYMM:
      concatenationGeneral<int8_t>();
      break;
    case OperandType::INT32:
      concatenationGeneral<int32_t>();
      break;
    case OperandType::INT64:
      concatenationGeneral<int64_t>();
      break;
    default:
      throw std::runtime_error("Concat: unsupported data type");
  }
}

} // namespace ops
} // namespace training
} // namespace backend
} // namespace onert
