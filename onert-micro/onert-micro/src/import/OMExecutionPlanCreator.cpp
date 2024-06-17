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

namespace
{

// Layers with trainable weights
// Note: needed not to store some layers with const intputs but it is not trainable (for example
// Reshape)
bool isTrainableWeights(const circle::OperatorCode *opcode)
{
  switch (opcode->builtin_code())
  {
    case circle::BuiltinOperator_FULLY_CONNECTED:
    case circle::BuiltinOperator_CONV_2D:
      return true;
    default:
      return false;
  }
}

} // namespace

/*
 * Create execution plan for forward graph
 * TODO: describe creation execution plan logic
 */
OMStatus OMExecutionPlanCreator::createExecutionPlan(core::OMRuntimeStorage &runtime_storage,
                                                     core::OMRuntimeContext &runtime_context,
                                                     core::memory::OMRuntimeAllocator &allocator,
                                                     const OMConfig &configs)
{
  bool keep_input = configs.keep_input;
  bool train_mode = configs.train_mode;

  std::vector<std::vector<uint16_t>> &alloc_plan = allocator.getAllocPlan();
  std::vector<std::vector<uint16_t>> &dealloc_plan = allocator.getDeallocPlan();

  using Lifetime = std::pair<int32_t, int32_t>;

  std::map<uint16_t, Lifetime> lifetimes;

  const reader::CircleOperators *operators = runtime_context.getCircleOperators();

  const size_t num_kernels = operators->size();

  uint32_t num_train_layers = configs.training_context.num_of_train_layers;
  if (train_mode and num_train_layers == 0)
    num_train_layers = num_kernels;

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
        if (kernel_type == Inplace or train_mode and index >= (num_kernels - num_train_layers))
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
      else if (train_mode and index >= (num_kernels - num_train_layers))
        lifetimes[output_index] = Lifetime(index, -1);
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

/*
 * Create execution plan for backward graph:
 * - Allocate memory for inputs for current op using the following rules:
 *   1) Don't allocate data for non const input tensor if it is last op for training (don't need to
 * backpropagate result) 2) Don't allocate data for const input tensor if is non trainable const 3)
 * Allocate data otherwise
 * - Deallocate memory for outputs for current op using the following rules:
 *   1) Deallocate all outputs tensors.
 */
OMStatus OMExecutionPlanCreator::createBackwardExecutionPlan(
  core::OMRuntimeStorage &runtime_storage, core::OMRuntimeContext &runtime_context,
  core::memory::OMRuntimeAllocator &allocator, const OMConfig &configs)
{
  bool keep_input = configs.keep_input;
  bool train_mode = configs.train_mode;
  assert(train_mode);
  if (train_mode == false)
    return UnknownError;

  std::vector<std::vector<uint16_t>> &alloc_plan = allocator.getAllocPlan();
  std::vector<std::vector<uint16_t>> &dealloc_plan = allocator.getDeallocPlan();

  using Lifetime = std::pair<int32_t, int32_t>;
  std::map<uint16_t, Lifetime> lifetimes;

  const reader::CircleOperators *operators = runtime_context.getCircleOperators();
  const uint32_t num_kernels = operators->size();

  uint32_t num_train_layers = configs.training_context.num_of_train_layers == 0
                                ? num_kernels
                                : configs.training_context.num_of_train_layers;
  auto graph_outputs = runtime_context.getCircleOutputs();

  for (const auto output_ind : *graph_outputs)
  {
    assert(lifetimes.count(output_ind) == 0);
    lifetimes[output_ind] = Lifetime(-1, 0);
  }

  uint32_t last_node_pos = std::min(num_kernels, num_train_layers);
  const auto *op_codes = runtime_context.getCircleOpcodes();
  for (int32_t index = 0; index < last_node_pos; ++index)
  {
    uint32_t cur_op_index = num_kernels - index - 1;
    auto *cur_op = operators->operator[](cur_op_index);

    uint32_t cur_opcode_index = cur_op->opcode_index();

    assert(cur_opcode_index < op_codes->size());

    const auto opcode = op_codes->operator[](cur_opcode_index);

    const auto *op_inputs = cur_op->inputs();
    const auto *op_outputs = cur_op->outputs();
    for (int32_t j = 0; j < op_inputs->size(); ++j)
    {
      const auto input_index = op_inputs->operator[](j);
      const auto is_const = runtime_context.isConstTensor(input_index);
      // Note: we dont need to allocate for last node and for empty tensor
      if (input_index == -1 or (is_const and not isTrainableWeights(opcode)) or
          ((index == last_node_pos - 1) and !is_const))
      {
        continue;
      }
      lifetimes[input_index] = {index, -1};
    }

    for (int32_t j = 0; j < op_outputs->size(); ++j)
    {
      const auto output_index = op_outputs->operator[](j);

      lifetimes.at(output_index).second = index;
    }
  }

  alloc_plan.assign(last_node_pos, std::vector<uint16_t>());
  dealloc_plan.assign(last_node_pos, std::vector<uint16_t>());

  for (const auto &item : lifetimes)
  {
    if (item.second.first != -1)
      alloc_plan[item.second.first].push_back(item.first);
    if (item.second.second != -1)
      dealloc_plan[item.second.second].push_back(item.first);
  }

  return Ok;
}
