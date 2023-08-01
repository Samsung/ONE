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
#include "TISOKernel.h"

#include "PALComparisons.h"

namespace luci_interpreter
{

namespace
{
// TODO: reduce code duplication with less
template <typename T>
void evalGeneric(const circle::Tensor *x, const circle::Tensor *y, const circle::Tensor *output,
                 BaseRuntimeGraph *runtime_graph)
{
  auto x_data = kernels::getTensorData<T>(runtime_graph->getDataByTensor(x));
  if (x_data == nullptr)
    x_data = kernels::getTensorData<T>(runtime_graph->getConstDataByTensor(x));

  assert(x_data != nullptr);

  auto y_data = kernels::getTensorData<T>(runtime_graph->getDataByTensor(y));
  if (y_data == nullptr)
    y_data = kernels::getTensorData<T>(runtime_graph->getConstDataByTensor(y));

  assert(y_data != nullptr);

  auto output_data = kernels::getTensorData<bool>(runtime_graph->getDataByTensor(output));

  luci_interpreter_pal::ComparisonParams op_params;
  op_params.is_broadcast = Tensor::num_elements(x) != Tensor::num_elements(y);

  const int64_t flat_size = kernels::getTensorShape(x).flatSize();
  luci_interpreter_pal::ComparisonNoScaling<T>(flat_size, x_data, y_data, output_data,
                                               luci_interpreter_pal::LessEqualFn);
}

} // namespace

void configure_kernel_CircleLessEqual(const circle::Operator *cur_op,
                                      BaseRuntimeGraph *runtime_graph)
{
  kernels::TISOKernel kernel(cur_op, runtime_graph);

  LUCI_INTERPRETER_CHECK(Tensor::element_type(kernel.input1()) ==
                         Tensor::element_type(kernel.input2()));
  LUCI_INTERPRETER_CHECK(Tensor::element_type(kernel.output()) == DataType::BOOL);
}

void execute_kernel_CircleLessEqual(const circle::Operator *cur_op, BaseRuntimeGraph *runtime_graph)
{
  kernels::TISOKernel kernel(cur_op, runtime_graph);

  switch (Tensor::element_type(kernel.input1()))
  {
    case DataType::S64:
      evalGeneric<int64_t>(kernel.input1(), kernel.input2(), kernel.output(), runtime_graph);
      break;
    case DataType::S32:
      evalGeneric<int32_t>(kernel.input1(), kernel.input2(), kernel.output(), runtime_graph);
      break;
#ifndef DIS_FLOAT
    case DataType::FLOAT32:
      evalGeneric<float>(kernel.input1(), kernel.input2(), kernel.output(), runtime_graph);
      break;
#endif // DIS_FLOAT
    default:
      assert(false && "Unsupported type.");
  }
}

} // namespace luci_interpreter
