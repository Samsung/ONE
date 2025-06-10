/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kernels/GRU.h"

#include "kernels/Utils.h"

#include "PALFullyConnected.h"
#include "PALGRU.h"

namespace luci_interpreter
{
namespace kernels
{
GRU::GRU(const Tensor *input, const Tensor *hidden_hidden, const Tensor *hidden_hidden_bias,
         const Tensor *hidden_input, const Tensor *hidden_input_bias, const Tensor *state,
         Tensor *output, const GRUParams &params)
  : KernelWithParams<GRUParams>(
      {input, hidden_hidden, hidden_hidden_bias, hidden_input, hidden_input_bias, state}, {output},
      params)
{
}

void GRU::configure()
{
  auto hidden_hidden_shape = getTensorShape(hidden_hidden());
  auto hidden_input_shape = getTensorShape(hidden_input());
  LUCI_INTERPRETER_CHECK(hidden_hidden_shape.Dims(0) == hidden_input_shape.Dims(0));

  output()->resize(state()->shape());

  LUCI_INTERPRETER_CHECK(input()->element_type() == output()->element_type());
}

void GRU::execute() const
{
  switch (input()->element_type())
  {
    case DataType::FLOAT32:
      evalFloat();
      break;
    default:
      throw std::runtime_error("luci-GRU Unsupported data type.");
  }
}

void GRU::evalFloat() const
{
  uint8_t *output_hidden_data;
  uint8_t *output_input_data;

  // allocate output datas above
  output_hidden_data = new uint8_t[getTensorShape(hidden_hidden()).FlatSize() * sizeof(float)];
  output_input_data = new uint8_t[getTensorShape(hidden_input()).FlatSize() * sizeof(float)];

  luci_interpreter_pal::GRU(
    getTensorData<float>(input()), getTensorData<float>(hidden_input()),
    getTensorData<float>(hidden_hidden()), getTensorData<float>(hidden_input_bias()),
    getTensorData<float>(hidden_hidden_bias()), getTensorData<float>(state()),
    getTensorData<float>(output()), reinterpret_cast<float *>(output_input_data),
    reinterpret_cast<float *>(output_hidden_data), getTensorShape(input()),
    getTensorShape(output()), getTensorShape(hidden_input()), getTensorShape(hidden_hidden()));

  delete[] output_hidden_data;
  delete[] output_input_data;
}

} // namespace kernels
} // namespace luci_interpreter
