/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/CumSum.h"

#include <tensorflow/lite/kernels/internal/optimized/optimized_ops.h>

#include "kernels/Utils.h"

namespace luci_interpreter
{
namespace kernels
{

CumSum::CumSum(const Tensor *input, const Tensor *axis, Tensor *output, const CumSumParams &params)
  : KernelWithParams<CumSumParams>({input, axis}, {output}, params)
{
}

void CumSum::configure()
{
  LUCI_INTERPRETER_CHECK(input()->element_type() == output()->element_type());
  LUCI_INTERPRETER_CHECK(input()->shape().num_dims() >= 1);

  LUCI_INTERPRETER_CHECK(axis()->element_type() == DataType::S32);
  LUCI_INTERPRETER_CHECK(axis()->shape().num_dims() == 0);

  output()->resize(input()->shape());
}

void CumSum::execute() const
{
  switch (input()->element_type())
  {
    case DataType::FLOAT32:
      tflite::optimized_ops::CumSum(getTensorData<float>(input()), getTensorShape(input()),
                                    *getTensorData<int32_t>(axis()), params().exclusive,
                                    params().reverse, getTensorData<float>(output()));
      break;
    default:
      throw std::runtime_error("Unsupported type.");
  }
}

} // namespace kernels
} // namespace luci_interpreter
