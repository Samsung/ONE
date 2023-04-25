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

#include "kernels/Dequantize.h"
#include "kernels/Utils.h"
#include "PALDequantize.h"

namespace luci_interpreter
{
namespace kernels
{

Dequantize::Dequantize(const Tensor *input, Tensor *output) : Kernel({input}, {output}) {}

void Dequantize::configure()
{
  LUCI_INTERPRETER_CHECK(input()->element_type() == DataType::S8 ||
                         input()->element_type() == DataType::U8 ||
                         input()->element_type() == DataType::S16);

  LUCI_INTERPRETER_CHECK(input()->scales().size() == 1);

  if (input()->element_type() == DataType::S16)
    LUCI_INTERPRETER_CHECK(input()->zero_point() == 0);

  LUCI_INTERPRETER_CHECK(output()->element_type() == DataType::FLOAT32);

  // TODO: enable it only if kernel with dynamic shapes
  output()->resize(input()->shape());
}

void Dequantize::execute() const
{
  tflite::DequantizationParams op_params;
  op_params.zero_point = input()->zero_point();
  op_params.scale = input()->scale();

  switch (input()->element_type())
  {
    case DataType::U8:
    {
      luci_interpreter_pal::Dequantize(op_params, getTensorShape(input()),
                                       getTensorData<uint8_t>(input()), getTensorShape(output()),
                                       getTensorData<float>(output()));
      break;
    }
    case DataType::S8:
    {
      luci_interpreter_pal::Dequantize(op_params, getTensorShape(input()),
                                       getTensorData<int8_t>(input()), getTensorShape(output()),
                                       getTensorData<float>(output()));
      break;
    }
    case DataType::S16:
    {
      luci_interpreter_pal::Dequantize(op_params, getTensorShape(input()),
                                       getTensorData<int16_t>(input()), getTensorShape(output()),
                                       getTensorData<float>(output()));
      break;
    }
    default:
      assert(false && "Unsupported type.");
  }
}

} // namespace kernels
} // namespace luci_interpreter
