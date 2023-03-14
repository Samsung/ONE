/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

#include <cassert>
#include <cstring>

namespace luci_interpreter
{

void configure_kernel_CircleReshape(const circle::Operator *cur_op, BaseRuntimeGraph *runtime_graph)
{
  // Do nothing
}

void execute_kernel_CircleReshape(const circle::Operator *cur_op, BaseRuntimeGraph *runtime_graph,
                                  bool is_inplace)
{
  const auto input_index = cur_op->inputs()->operator[](0);
  const auto output_index = cur_op->outputs()->operator[](0);

  assert(input_index != -1);
  assert(output_index != -1);

  const auto input = runtime_graph->getCircleTensorByIndex(input_index);
  const auto output = runtime_graph->getCircleTensorByIndex(output_index);

  if (is_inplace)
  {
    runtime_graph->makeInplaceOperation(input, output);
    return;
  }

  const auto input_data = (runtime_graph->getDataByTensor(input));
  auto output_data = (runtime_graph->getDataByTensor(output));

  assert(input_data != nullptr);
  assert(output_data != nullptr);

  const size_t element_size = getDataTypeSize(Tensor::element_type(input));
  const int32_t num_elements = Tensor::num_elements(input);
  std::memcpy(output_data, input_data, num_elements * element_size);
}

} // namespace luci_interpreter
