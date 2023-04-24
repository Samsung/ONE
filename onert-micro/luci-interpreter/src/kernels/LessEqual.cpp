/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/LessEqual.h"
#include "kernels/Utils.h"

#include <tensorflow/lite/kernels/internal/reference/comparisons.h>

namespace luci_interpreter
{

namespace kernels
{

LessEqual::LessEqual(const Tensor *x, const Tensor *y, Tensor *output) : Kernel({x, y}, {output}) {}

void LessEqual::configure()
{
  LUCI_INTERPRETER_CHECK(x()->element_type() == y()->element_type());
  LUCI_INTERPRETER_CHECK(output()->element_type() == DataType::BOOL);

  if (x()->element_type() == DataType::U8)
  {
    quantizeMultiplierSmallerThanOneExp(x()->scale(), &_x_multiplier, &_x_shift);
    quantizeMultiplierSmallerThanOneExp(y()->scale(), &_y_multiplier, &_y_shift);
  }
  // TODO: enable it only if kernel with dynamic shapes
  output()->resize(calculateShapeForBroadcast(x()->shape(), y()->shape()));
}

void LessEqual::execute() const
{
  switch (x()->element_type())
  {
    case DataType::FLOAT32:
      evalFloat();
      break;
    case DataType::S64:
      evalInteger<int64_t>();
      break;
    case DataType::S32:
      evalInteger<int32_t>();
      break;
    case DataType::U8:
      evalQuantized();
      break;
    default:
      assert(false && "Unsupported type.");
  }
}

void LessEqual::evalFloat() const
{
  const auto x_data = getTensorData<float>(x());
  const auto y_data = getTensorData<float>(y());
  auto output_data = getTensorData<bool>(output());

  tflite::ComparisonParams op_params;
  op_params.is_broadcast = x()->shape() != y()->shape();

  if (op_params.is_broadcast)
  {
    tflite::reference_ops::Broadcast4DSlowLessEqual(op_params, getTensorShape(x()), x_data,
                                                    getTensorShape(y()), y_data,
                                                    getTensorShape(output()), output_data);
  }
  else
  {
    tflite::reference_ops::LessEqual(op_params, getTensorShape(x()), x_data, getTensorShape(y()),
                                     y_data, getTensorShape(output()), output_data);
  }
}

template <typename T> void LessEqual::evalInteger() const
{
  const auto x_data = getTensorData<T>(x());
  const auto y_data = getTensorData<T>(y());
  auto output_data = getTensorData<bool>(output());

  tflite::ComparisonParams op_params;
  op_params.is_broadcast = x()->shape() != y()->shape();

  if (op_params.is_broadcast)
  {
    tflite::reference_ops::Broadcast4DSlowLessEqualNoScaling(op_params, getTensorShape(x()), x_data,
                                                             getTensorShape(y()), y_data,
                                                             getTensorShape(output()), output_data);
  }
  else
  {
    tflite::reference_ops::LessEqualNoScaling(op_params, getTensorShape(x()), x_data,
                                              getTensorShape(y()), y_data, getTensorShape(output()),
                                              output_data);
  }
}

void LessEqual::evalQuantized() const
{
  const auto x_data = getTensorData<uint8_t>(x());
  const auto y_data = getTensorData<uint8_t>(y());
  auto output_data = getTensorData<bool>(output());

  tflite::ComparisonParams op_params;
  op_params.left_shift = 8;
  op_params.input1_offset = -x()->zero_point(); // Note the '-'
  op_params.input1_shift = _x_shift;
  op_params.input1_multiplier = _x_multiplier;
  op_params.input2_offset = -y()->zero_point(); // Note the '-'
  op_params.input2_shift = _y_shift;
  op_params.input2_multiplier = _y_multiplier;
  op_params.is_broadcast = x()->shape() != y()->shape();

  if (op_params.is_broadcast)
  {
    tflite::reference_ops::Broadcast4DSlowLessEqualWithScaling(
      op_params, getTensorShape(x()), x_data, getTensorShape(y()), y_data, getTensorShape(output()),
      output_data);
  }
  else
  {
    tflite::reference_ops::LessEqualWithScaling(op_params, getTensorShape(x()), x_data,
                                                getTensorShape(y()), y_data,
                                                getTensorShape(output()), output_data);
  }
}

} // namespace kernels
} // namespace luci_interpreter
