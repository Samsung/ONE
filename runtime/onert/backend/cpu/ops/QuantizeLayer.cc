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

#include "OperationUtils.h"
#include "QuantizeLayer.h"

#include <cker/operation/Dequantize.h>
#include <cker/operation/Erf.h>
#include <cker/operation/Exp.h>
#include <cker/operation/LogicalNot.h>
#include <cker/operation/Quantize.h>
#include <cker/operation/Round.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
{
template <typename InputT, typename OutputT>
void affineQuantize(const IPortableTensor *input, IPortableTensor *output)
{
  nnfw::cker::Quantize(getTensorShape(input), reinterpret_cast<const InputT *>(input->buffer()),
                       getTensorShape(output), reinterpret_cast<OutputT *>(output->buffer()),
                       output->data_scale(), output->data_offset());
}

void QuantizeLayer::configure(const IPortableTensor *input, IPortableTensor *output)
{
  assert(input != nullptr);
  assert(output != nullptr);

  _input = input;
  _output = output;

  if ((_input->data_type() == OperandType::FLOAT32))
  {
    // DO NOTHING
  }
  else if ((input->data_type() == OperandType::QUANT_UINT8_ASYMM) &&
           (output->data_type() == OperandType::QUANT_INT8_ASYMM))
  {
    const double effective_output_scale =
      static_cast<double>(input->data_scale()) / static_cast<double>(output->data_scale());
    QuantizeMultiplier(effective_output_scale, &_output_multiplier, &_output_shift);
  }
  else
  {
    throw std::runtime_error{"Quantize: Unsupported  data type"};
  }
}

void QuantizeLayer::run()
{
  if ((_input->data_type() == OperandType::FLOAT32))
  {
    affineQuantize<float, uint8_t>(_input, _output);
  }
  else if ((_input->data_type() == OperandType::QUANT_UINT8_ASYMM) &&
           (_output->data_type() == OperandType::QUANT_INT8_ASYMM))
  {
    nnfw::cker::Requantize<uint8_t, int8_t>(
      reinterpret_cast<const uint8_t *>(_input->buffer()),
      MatchingFlatSize(getTensorShape(_input), getTensorShape(_output)), _output_multiplier,
      _output_shift, _input->data_offset(), _output->data_offset(),
      reinterpret_cast<int8_t *>(_output->buffer()));
  }
  else
  {
    throw std::runtime_error{"Quantize: Unsupported  data type"};
  }
}

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert
