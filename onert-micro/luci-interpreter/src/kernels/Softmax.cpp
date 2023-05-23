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

#include "Builders.h"
#include "kernels/Utils.h"
#include "SISOKernel.h"

#include <tensorflow/lite/kernels/internal/reference/softmax.h>

namespace luci_interpreter
{

namespace
{

#ifndef DIS_FLOAT
void evalFloat(const circle::Tensor *input, const circle::Tensor *output,
               const circle::SoftmaxOptions *options, BaseRuntimeGraph *runtime_graph)
{
  const auto *input_data = runtime_graph->getDataByTensor(input);
  auto *output_data = runtime_graph->getDataByTensor(output);

  tflite::SoftmaxParams op_params{};
  op_params.beta = options->beta();

  tflite::reference_ops::Softmax(
    op_params, kernels::getTensorShape(input), kernels::getTensorData<float>(input_data),
    kernels::getTensorShape(output), kernels::getTensorData<float>(output_data));
}
#endif // DIS_FLOAT

} // namespace

void configure_kernel_CircleSoftmax(const circle::Operator *cur_op, BaseRuntimeGraph *runtime_graph)
{
  kernels::SISOKernel kernel(cur_op, runtime_graph);

  LUCI_INTERPRETER_CHECK(Tensor::element_type(kernel.input()) ==
                         Tensor::element_type(kernel.output()));
  LUCI_INTERPRETER_CHECK(Tensor::num_dims(kernel.input()) >= 1);

#ifndef DIS_QUANT
  if (Tensor::element_type(kernel.input()) == DataType::U8 ||
      Tensor::element_type(kernel.input()) == DataType::S8)
  {
    LUCI_INTERPRETER_CHECK(Tensor::element_type(kernel.input()) == DataType::S8 ||
                           Tensor::zero_point(kernel.output()) == 0);
    LUCI_INTERPRETER_CHECK(Tensor::element_type(kernel.input()) == DataType::U8 ||
                           Tensor::zero_point(kernel.output()) ==
                             std::numeric_limits<int8_t>::min());
  }
#endif
}

void execute_kernel_CircleSoftmax(const circle::Operator *cur_op, BaseRuntimeGraph *runtime_graph)
{
  kernels::SISOKernel kernel(cur_op, runtime_graph);

  const auto *options = cur_op->builtin_options_as_SoftmaxOptions();

  switch (Tensor::element_type(kernel.input()))
  {
#ifndef DIS_FLOAT
    case DataType::FLOAT32:
      evalFloat(kernel.input(), kernel.output(), options, runtime_graph);
      break;
#endif // DIS_FLOAT
    default:
      assert(false && "Unsupported type.");
  }
}

} // namespace luci_interpreter
