/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "import/OMExecutionPlanCreator.h"

#include <map>

using namespace onert_micro::core;
using namespace onert_micro::import;
using namespace onert_micro;

OMStatus OMExecutionPlanCreator::createExecutionPlan(core::OMRuntimeStorage &runtime_storage,
                                                     core::OMRuntimeContext &runtime_context,
                                                     core::memory::OMRuntimeAllocator &allocator,
                                                     const OMConfig &configs,
                                                     const std::unordered_set<uint16_t> &saved_tensors_indexes)
{
  bool keep_input = configs.keep_input;

  std::vector<std::vector<uint16_t>> &alloc_plan = allocator.getAllocPlan();
  std::vector<std::vector<uint16_t>> &dealloc_plan = allocator.getDeallocPlan();

  using Lifetime = std::pair<int32_t, int32_t>;

  std::map<uint16_t, Lifetime> lifetimes;

  const reader::CircleOperators *operators = runtime_context.getCircleOperators();

  const size_t num_kernels = operators->size();

  if (not keep_input)
  {
    auto graph_inputs = runtime_context.getCircleInputs();
    for (const auto input_ind : *graph_inputs)
    {
      assert(lifetimes.count(input_ind) == 0);
      lifetimes[input_ind] = Lifetime(-1, 0);
    }
  }

  for (int32_t index = 0; index < num_kernels; ++index)
  {
    auto *cur_op = operators->operator[](index);

    const auto *op_inputs = cur_op->inputs();
    const auto *op_outputs = cur_op->outputs();
    auto kernel_type = runtime_storage.getKernelType(index);
    for (int32_t j = 0; j < op_inputs->size(); ++j)
    {
      const auto input_index = op_inputs->operator[](j);

      if (input_index == -1)
        continue;

      // Pass constant tensors
      if (runtime_context.isConstTensor(input_index))
        continue;

      if (lifetimes.count(input_index) > 0)
      {
        // TODO: replace saved_tesnors_indexes with new kernel type
        if ((kernel_type == Inplace and j == 0) or saved_tensors_indexes.find(input_index) != saved_tensors_indexes.end())
          lifetimes.at(input_index).second = -1;
        else
          lifetimes.at(input_index).second = index;
      }
    }

    for (int32_t j = 0; j < op_outputs->size(); ++j)
    {
      const auto output_index = op_outputs->operator[](j);

      if (kernel_type == Inplace)
        lifetimes[output_index] = Lifetime(-1, index);
      else
        lifetimes[output_index] = Lifetime(index, index);
    }
  }
  auto graph_outputs = runtime_context.getCircleOutputs();
  for (const auto output_ind : *graph_outputs)
  {
    if (lifetimes.count(output_ind) > 0)
      lifetimes.at(output_ind).second = static_cast<int32_t>(num_kernels);
  }

  alloc_plan.assign(num_kernels, std::vector<uint16_t>());
  dealloc_plan.assign(num_kernels + 1, std::vector<uint16_t>());

  for (const auto &item : lifetimes)
  {
    if (item.second.first != -1)
      alloc_plan[item.second.first].push_back(item.first);
    if (item.second.second != -1)
      dealloc_plan[item.second.second].push_back(item.first);
  }

  return Ok;
}
