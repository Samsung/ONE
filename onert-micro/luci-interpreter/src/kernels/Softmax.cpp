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

#include "Builders.h"
#include "kernels/Utils.h"

#include <tensorflow/lite/kernels/internal/reference/softmax.h>
#include "PALSoftmax.h"

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

#ifndef DIS_QUANT
template <typename T>
void evalQuantized(const circle::Tensor *input, const circle::Tensor *output,
                   const circle::SoftmaxOptions *options, BaseRuntimeGraph *runtime_graph)
{
  // TODO: Enable it
  assert(false && "Not impl yet");

  const auto *input_data = runtime_graph->getDataByTensor(input);
  auto *output_data = runtime_graph->getDataByTensor(output);

  tflite::SoftmaxParams op_params{};

  luci_interpreter_pal::InitializeParams(&op_params, Tensor::scale(input), options->beta());
  luci_interpreter_pal::Softmax(
    op_params, kernels::getTensorShape(input), kernels::getTensorData<T>(input_data),
    kernels::getTensorShape(output), kernels::getTensorData<T>(output_data));
}
#endif

} // namespace

void configure_kernel_CircleSoftmax(const circle::Operator *cur_op, BaseRuntimeGraph *runtime_graph)
{
  const auto input_index = cur_op->inputs()->operator[](0);
  const auto output_index = cur_op->outputs()->operator[](0);

  assert(input_index != -1);
  assert(output_index != -1);

  const auto input = runtime_graph->getCircleTensorByIndex(input_index);
  auto output = runtime_graph->getCircleTensorByIndex(output_index);

  assert(input != nullptr);
  assert(output != nullptr);

  LUCI_INTERPRETER_CHECK(Tensor::element_type(input) == Tensor::element_type(output));
  LUCI_INTERPRETER_CHECK(Tensor::num_dims(input) >= 1);

#ifndef DIS_QUANT
  if (Tensor::element_type(input) == DataType::U8 || Tensor::element_type(input) == DataType::S8)
  {
    LUCI_INTERPRETER_CHECK(Tensor::element_type(input) == DataType::S8 ||
                           Tensor::zero_point(output) == 0);
    LUCI_INTERPRETER_CHECK(Tensor::element_type(input) == DataType::U8 ||
                           Tensor::zero_point(output) == std::numeric_limits<int8_t>::min());
  }
#endif
}

void execute_kernel_CircleSoftmax(const circle::Operator *cur_op, BaseRuntimeGraph *runtime_graph,
                                  bool)
{
  const auto input_index = cur_op->inputs()->operator[](0);
  const auto output_index = cur_op->outputs()->operator[](0);

  assert(input_index != -1);
  assert(output_index != -1);

  const auto input = runtime_graph->getCircleTensorByIndex(input_index);
  auto output = runtime_graph->getCircleTensorByIndex(output_index);

  assert(input != nullptr);
  assert(output != nullptr);

  const auto *options = cur_op->builtin_options_as_SoftmaxOptions();

  switch (Tensor::element_type(input))
  {
#ifndef DIS_FLOAT
    case DataType::FLOAT32:
      evalFloat(input, output, options, runtime_graph);
      break;
#endif // DIS_FLOAT
#ifndef DIS_QUANT
    case DataType::S8:
      evalQuantized<int8_t>(input, output, options, runtime_graph);
      break;
    case DataType::U8:
      evalQuantized<uint8_t>(input, output, options, runtime_graph);
      break;
#endif // DIS_QUANT
    default:
      assert(false && "Unsupported type.");
  }
}

} // namespace luci_interpreter
