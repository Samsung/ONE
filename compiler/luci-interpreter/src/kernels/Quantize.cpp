/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/Quantize.h"
#include "kernels/Utils.h"
#include "PALQuantize.h"

namespace luci_interpreter
{
namespace kernels
{

namespace
{
void compute_shift_multiplier(const double input_scale, const double output_scale,
                              int32_t *multiplier, int32_t *shift)
{
  const double effective_output_scale = input_scale / output_scale;
  quantizeMultiplier(effective_output_scale, multiplier, shift);
}

} // namespace

Quantize::Quantize(const Tensor *input, Tensor *output) : Kernel({input}, {output}) {}

void Quantize::configure()
{
  if (input()->element_type() == loco::DataType::FLOAT32)
  {
    LUCI_INTERPRETER_CHECK(output()->element_type() == loco::DataType::U8 ||
                           output()->element_type() == loco::DataType::S8 ||
                           output()->element_type() == loco::DataType::S16);
  }
  else if (input()->element_type() == loco::DataType::S16)
  {
    LUCI_INTERPRETER_CHECK(output()->element_type() == loco::DataType::S8 ||
                           output()->element_type() == loco::DataType::S16 ||
                           output()->element_type() == loco::DataType::S32);
    if (output()->element_type() == loco::DataType::S16)
    {
      LUCI_INTERPRETER_CHECK(input()->zero_point() == 0 && output()->zero_point() == 0);
    }
  }
  else if (input()->element_type() == loco::DataType::S8 ||
           input()->element_type() == loco::DataType::U8)
  {
    LUCI_INTERPRETER_CHECK(output()->element_type() == loco::DataType::S8 ||
                           output()->element_type() == loco::DataType::U8);
  }
  else
  {
    throw std::runtime_error("Unsupported type.");
  }

  output()->resize(input()->shape());
}

void Quantize::execute() const
{
  switch (input()->element_type())
  {
    case loco::DataType::FLOAT32:
    {
      tflite::QuantizationParams op_params;
      op_params.zero_point = output()->zero_point();
      op_params.scale = output()->scale();
      const auto input_data = getTensorData<float>(input());

      switch (output()->element_type())
      {
        case loco::DataType::S8:
        {
          luci_interpreter_pal::Quantize(op_params, getTensorShape(input()), input_data,
                                         getTensorShape(output()), getTensorData<int8_t>(output()));
          break;
        }
        case loco::DataType::U8:
        {
          luci_interpreter_pal::Quantize(op_params, getTensorShape(input()), input_data,
                                         getTensorShape(output()),
                                         getTensorData<uint8_t>(output()));
          break;
        }
        case loco::DataType::S16:
        {
          luci_interpreter_pal::Quantize(op_params, getTensorShape(input()), input_data,
                                         getTensorShape(output()),
                                         getTensorData<int16_t>(output()));
          break;
        }
        default:
          throw std::runtime_error("Unsupported type.");
      }
      break;
    }
    case loco::DataType::S16:
    {
      int32_t multiplier;
      int32_t shift;

      compute_shift_multiplier(static_cast<double>(input()->scale()),
                               static_cast<double>(output()->scale()), &multiplier, &shift);

      const auto input_shape = getTensorShape(input());
      const auto output_shape = getTensorShape(output());

      const auto input_data = getTensorData<int16_t>(input());

      const auto size = tflite::MatchingFlatSize(input_shape, output_shape);

      switch (output()->element_type())
      {
        case loco::DataType::S8:
        {
          luci_interpreter_pal::Requantize(input_data, size, multiplier, shift,
                                           input()->zero_point(), output()->zero_point(),
                                           getTensorData<int8_t>(output()));
          break;
        }
        case loco::DataType::S16:
        {
          luci_interpreter_pal::Requantize(input_data, size, multiplier, shift,
                                           input()->zero_point(), output()->zero_point(),
                                           getTensorData<int16_t>(output()));
          break;
        }
        case loco::DataType::S32:
        {
          luci_interpreter_pal::Requantize(input_data, size, multiplier, shift,
                                           input()->zero_point(), output()->zero_point(),
                                           getTensorData<int32_t>(output()));
          break;
        }
        default:
          throw std::runtime_error("Unsupported type.");
      }
      break;
    }
    case loco::DataType::S8:
    {
      int32_t multiplier;
      int32_t shift;

      compute_shift_multiplier(static_cast<double>(input()->scale()),
                               static_cast<double>(output()->scale()), &multiplier, &shift);

      const auto input_data = getTensorData<int8_t>(input());

      const auto input_shape = getTensorShape(input());
      const auto output_shape = getTensorShape(output());

      const auto size = tflite::MatchingFlatSize(input_shape, output_shape);

      switch (output()->element_type())
      {
        case loco::DataType::S8:
        {
          luci_interpreter_pal::Requantize(input_data, size, multiplier, shift,
                                           input()->zero_point(), output()->zero_point(),
                                           getTensorData<int8_t>(output()));
          break;
        }
        case loco::DataType::U8:
        {
          luci_interpreter_pal::Requantize(input_data, size, multiplier, shift,
                                           input()->zero_point(), output()->zero_point(),
                                           getTensorData<uint8_t>(output()));
          break;
        }
        default:
          throw std::runtime_error("Unsupported type.");
      }
      break;
    }
    case loco::DataType::U8:
    {
      int32_t multiplier;
      int32_t shift;

      compute_shift_multiplier(static_cast<double>(input()->scale()),
                               static_cast<double>(output()->scale()), &multiplier, &shift);

      const auto input_data = getTensorData<uint8_t>(input());

      const auto input_shape = getTensorShape(input());
      const auto output_shape = getTensorShape(output());

      const auto size = tflite::MatchingFlatSize(input_shape, output_shape);

      switch (output()->element_type())
      {
        case loco::DataType::S8:
        {
          luci_interpreter_pal::Requantize(input_data, size, multiplier, shift,
                                           input()->zero_point(), output()->zero_point(),
                                           getTensorData<int8_t>(output()));
          break;
        }
        case loco::DataType::U8:
        {
          luci_interpreter_pal::Requantize(input_data, size, multiplier, shift,
                                           input()->zero_point(), output()->zero_point(),
                                           getTensorData<uint8_t>(output()));
          break;
        }
        default:
          throw std::runtime_error("Unsupported type.");
      }
      break;
    }
    default:
      throw std::runtime_error("Unsupported type.");
  }
}

} // namespace kernels
} // namespace luci_interpreter
