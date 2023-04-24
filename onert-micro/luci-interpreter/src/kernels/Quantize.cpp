/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

template <typename input_dtype> void call_requantize(const Tensor *input, Tensor *output)
{
  int32_t multiplier;
  int shift;

  const double effective_output_scale = input->scale() / output->scale();
  quantizeMultiplier(effective_output_scale, &multiplier, &shift);

  const auto input_shape = getTensorShape(input);
  const auto output_shape = getTensorShape(output);
  const auto size = tflite::MatchingFlatSize(input_shape, output_shape);

  const auto input_data = getTensorData<input_dtype>(input);

  switch (output->element_type())
  {
    case DataType::S8:
      luci_interpreter_pal::Requantize(input_data, size, multiplier, shift, input->zero_point(),
                                       output->zero_point(), getTensorData<int8_t>(output));
      break;
    case DataType::U8:
      luci_interpreter_pal::Requantize(input_data, size, multiplier, shift, input->zero_point(),
                                       output->zero_point(), getTensorData<uint8_t>(output));
      break;
    case DataType::S16:
      luci_interpreter_pal::Requantize(input_data, size, multiplier, shift, input->zero_point(),
                                       output->zero_point(), getTensorData<int16_t>(output));
      break;
    default:
      assert(false && "Unsupported quantized type, yet!");
  }
}

} // namespace

Quantize::Quantize(const Tensor *input, Tensor *output) : Kernel({input}, {output}) {}

void Quantize::configure()
{

  if (input()->element_type() == DataType::S16)
    LUCI_INTERPRETER_CHECK(input()->zero_point() == 0);

  switch (input()->element_type())
  {
    case DataType::FLOAT32:
    {
      LUCI_INTERPRETER_CHECK(output()->element_type() == DataType::U8 ||
                             output()->element_type() == DataType::S8 ||
                             output()->element_type() == DataType::S16);
      break;
    }
    case DataType::S16:
    case DataType::S8:
    case DataType::U8:
    {
      LUCI_INTERPRETER_CHECK(output()->element_type() == DataType::S8 ||
                             output()->element_type() == DataType::U8 ||
                             output()->element_type() == DataType::S16);
      if (output()->element_type() == DataType::S16)
      {
        LUCI_INTERPRETER_CHECK(output()->zero_point() == 0);
      }
      break;
    }
    default:
      assert(false && "Unsupported type");
  }
  // TODO: enable it only if kernel with dynamic shapes
  output()->resize(input()->shape());
}

void Quantize::execute() const
{
  switch (input()->element_type())
  {
    case DataType::FLOAT32:
    {
      tflite::QuantizationParams op_params;
      op_params.zero_point = output()->zero_point();
      op_params.scale = output()->scale();
      const auto input_data = getTensorData<float>(input());

      switch (output()->element_type())
      {
        case DataType::S8:
        {
          luci_interpreter_pal::Quantize(op_params, getTensorShape(input()), input_data,
                                         getTensorShape(output()), getTensorData<int8_t>(output()));
          break;
        }
        case DataType::U8:
        {
          luci_interpreter_pal::Quantize(op_params, getTensorShape(input()), input_data,
                                         getTensorShape(output()),
                                         getTensorData<uint8_t>(output()));
          break;
        }
        case DataType::S16:
        {
          luci_interpreter_pal::Quantize(op_params, getTensorShape(input()), input_data,
                                         getTensorShape(output()),
                                         getTensorData<int16_t>(output()));
          break;
        }
        default:
          assert(false && "Unsupported type.");
      }
      break;
    }
    case DataType::S16:
    {
      call_requantize<int16_t>(input(), output());
      break;
    }
    case DataType::S8:
    {
      call_requantize<int8_t>(input(), output());
      break;
    }
    case DataType::U8:
    {
      call_requantize<uint8_t>(input(), output());
      break;
    }
    default:
      assert(false && "Unsupported type.");
  }
}

} // namespace kernels
} // namespace luci_interpreter
