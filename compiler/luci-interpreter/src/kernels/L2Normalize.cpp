/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/L2Normalize.h"
#include "kernels/Utils.h"

#include "PALL2Normalize.h"

#include <stdexcept>

namespace luci_interpreter
{

namespace kernels
{

L2Normalize::L2Normalize(const Tensor *input, Tensor *output, const L2NormParams &params)
  : KernelWithParams<L2NormParams>({input}, {output}, params)
{
}

void L2Normalize::configure()
{
  LUCI_INTERPRETER_CHECK(input()->shape().num_dims() <= 4);
  LUCI_INTERPRETER_CHECK(output()->element_type() == DataType::FLOAT32 ||
                         output()->element_type() == DataType::U8);
  LUCI_INTERPRETER_CHECK(input()->element_type() == output()->element_type());
  if (output()->element_type() == DataType::U8)
  {
    LUCI_INTERPRETER_CHECK(output()->scale() == (1. / 128.));
    LUCI_INTERPRETER_CHECK(output()->zero_point() == 128);
  }
  LUCI_INTERPRETER_CHECK(params().activation == Activation::NONE);
  output()->resize(input()->shape());
}

void L2Normalize::execute() const
{
  switch (output()->element_type())
  {
    case DataType::FLOAT32:
      eval<float>(0);
      break;
    case DataType::U8:
      eval<uint8_t>(input()->zero_point());
      break;
    default:
      throw std::runtime_error("luci-intp L2Normalize Unsupported type.");
  }
}

template <typename T> void L2Normalize::eval(int32_t zero_point) const
{
  tflite::L2NormalizationParams op_params{};
  op_params.input_zero_point = zero_point;
  luci_interpreter_pal::L2Normalization(op_params, getTensorShape(input()),
                                        getTensorData<T>(input()), getTensorShape(output()),
                                        getTensorData<T>(output()));
}

} // namespace kernels
} // namespace luci_interpreter
