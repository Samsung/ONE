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

#include "kernels/RoPE.h"

#include "kernels/Utils.h"

namespace luci_interpreter
{
namespace kernels
{

RoPE::RoPE(const Tensor *input, const Tensor *sin_table, const Tensor *cos_table, Tensor *output,
           const RoPEParams &params)
  : KernelWithParams<RoPEParams>({input, sin_table, cos_table}, {output}, params)
{
}

void RoPE::configure()
{
  LUCI_INTERPRETER_CHECK(input()->shape().num_dims() == 4);
  LUCI_INTERPRETER_CHECK(sin_table()->shape().dim(3) == input()->shape().dim(3));
  LUCI_INTERPRETER_CHECK(cos_table()->shape().dim(3) == input()->shape().dim(3));

  LUCI_INTERPRETER_CHECK(params().mode == RoPEMode::GPT_NEOX);

  output()->resize(input()->shape());
}

void RoPE::execute() const
{
  switch (input()->element_type())
  {
    case DataType::FLOAT32:
      evalFloat();
      break;
    default:
      throw std::runtime_error("luci-rope Unsupported data type.");
  }
}

void RoPE::evalFloat() const
{
  const auto input_shape = getTensorShape(input());
  const auto sin_table_shape = getTensorShape(sin_table());
  const auto cos_table_shape = getTensorShape(cos_table());
  auto output_shape = getTensorShape(output());

  const float *input_data = getTensorData<float>(input());
  const float *sin_table_data = getTensorData<float>(sin_table());
  const float *cos_table_data = getTensorData<float>(cos_table());
  float *output_data = getTensorData<float>(output());

  if (params().mode == RoPEMode::GPT_NEOX)
  {
    const int32_t i0_n = input_shape.Dims(0);
    const int32_t i1_n = input_shape.Dims(1); // multihead
    const int32_t i2_n = input_shape.Dims(2);
    const int32_t i3_n = input_shape.Dims(3); // head

    for (int32_t i0 = 0; i0 < i0_n; ++i0)
    {
      for (int32_t i1 = 0; i1 < i1_n; ++i1)
      {
        for (int32_t i2 = 0; i2 < i2_n; ++i2)
        {
          for (int32_t i3 = 0; i3 < i3_n / 2; ++i3)
          {
            const int32_t offset = tflite::Offset(input_shape, i0, i1, i2, i3);
            const float x0 = input_data[offset];
            const float x1 = input_data[offset + i3_n / 2];

            output_data[offset] = x0 * cos_table_data[i3] - x1 * sin_table_data[i3];
            output_data[offset + i3_n / 2] =
              x0 * sin_table_data[i3 + i3_n / 2] + x1 * cos_table_data[i3 + i3_n / 2];
          }
        }
      }
    }
  }
  else
    throw std::runtime_error("luci-intp RoPE unsupported mode.");
}

} // namespace kernels
} // namespace luci_interpreter
