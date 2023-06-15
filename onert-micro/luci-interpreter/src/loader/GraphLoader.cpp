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

#include "loader/GraphLoader.h"

namespace luci_interpreter
{
namespace
{

bool isInplaceOperation(const circle::BuiltinOperator &op)
{
  switch (op)
  {
    case circle::BuiltinOperator_LOGISTIC:
    case circle::BuiltinOperator_RESHAPE:
    case circle::BuiltinOperator_EXPAND_DIMS:
    case circle::BuiltinOperator_TANH:
    case circle::BuiltinOperator_ADD:
    case circle::BuiltinOperator_MUL:
    case circle::BuiltinOperator_SUB:
    case circle::BuiltinOperator_WHILE:
      return true;
    default:
      return false;
  }
}

bool isSingleUsageOfTensor(CircleReader *reader, const int32_t tensor_index)
{
  uint32_t usage_count = 0;

  const auto operators = reader->operators();
  for (uint32_t i = 0; i < operators.size(); ++i)
  {
    const auto *op = operators.at(i);
    assert(op != nullptr);

    const auto *op_inputs = op->inputs();
    for (int32_t j = 0; j < op_inputs->size(); ++j)
    {
      const auto input_index = op_inputs->operator[](j);
      if (input_index == tensor_index)
      {
        if (++usage_count > 1)
          return false;
      }
    }
  }

  // Let's check that it is not graph output
  if (usage_count == 1)
  {
    const auto &outputs_indexes = reader->outputs();
    bool is_graph_output = (std::find(outputs_indexes.begin(), outputs_indexes.end(),
                                      tensor_index) != outputs_indexes.end());
    if (is_graph_output)
      return false;
  }

  return true;
}

void buildMemoryUsageVector(CircleReader *reader, std::vector<int> &operations_mem_usage)
{
  const auto operators = reader->operators();
  const auto graph_outputs = reader->outputs();
  // Vector of memory
  // memory - needed memory for intermediate tensors to run operation at position = index
  for (uint32_t i = 0; i < operators.size(); ++i)
  {
    const auto *op = operators.at(i);
    assert(op != nullptr);

    const auto *op_inputs = op->inputs();
    const auto *op_outputs = op->outputs();

    // Pass operations with multiple outputs
    if (op_outputs->size() > 1)
      continue;

    const auto output_index = op_outputs->operator[](0);

    // Let's check that output is not a graph output tensor
    if (std::find(graph_outputs.begin(), graph_outputs.end(), output_index) != graph_outputs.end())
      continue;

    int cur_input_mem_overhead = 0;

    for (const auto input_idx : *op_inputs)
    {
      if (input_idx == -1)
        continue;
      const auto cur_input_tensor = reader->tensors()[input_idx];
      const auto input_type = Tensor::element_type(cur_input_tensor);
      if (Tensor::is_constant_tensor(reader, cur_input_tensor) or input_type != DataType::FLOAT32)
        continue;

      cur_input_mem_overhead += Tensor::num_elements(cur_input_tensor) * sizeof(input_type);
    }

    // Pass this operation
    if (cur_input_mem_overhead == 0)
      continue;

    int cur_output_mem_overhead = 0;
    for (const auto output_idx : *op_outputs)
    {
      if (output_idx == -1)
        continue;
      const auto cur_output_tensor = reader->tensors()[output_idx];
      const auto output_type = Tensor::element_type(cur_output_tensor);
      if (Tensor::is_constant_tensor(reader, cur_output_tensor) or output_type != DataType::FLOAT32)
        continue;

      cur_output_mem_overhead += Tensor::num_elements(cur_output_tensor) * sizeof(output_type);
    }

    // Pass this operation
    if (cur_output_mem_overhead == 0)
      continue;

    operations_mem_usage.push_back(cur_input_mem_overhead + cur_output_mem_overhead);
  }
}

void addOperationsStatusForRange(int start, int end, CircleReader *reader,
                                 RuntimeGraph *runtime_graph)
{
  const auto operators = reader->operators();
  {
    auto *start_op = operators.at(start);
    assert(start_op != nullptr);

    // Downgrade start index while this op is inplace
    while (runtime_graph->is_inplace_op(start_op) and start != end)
    {
      start++;
      start_op = operators.at(start);
      assert(start_op != nullptr);
    }
  }

  {
    auto *end_op = operators.at(end);
    assert(end_op != nullptr);

    // Downgrade start index while this op is inplace
    while (runtime_graph->is_inplace_op(end_op) and start != end)
    {
      end--;
      end_op = operators.at(end);
      assert(end_op != nullptr);
    }
  }

  // Nothing to do
  if (start >= end)
    return;

  // Add START status to operator
  {
    auto *start_op = operators.at(start);
    assert(start_op != nullptr);

    runtime_graph->addOperatorStatus(start_op, OperationGraphStatus::START);
  }

  // Add END status to operator
  {
    auto *end_op = operators.at(end);
    assert(end_op != nullptr);

    runtime_graph->addOperatorStatus(end_op, OperationGraphStatus::END);
  }

  // Add MIDDLE status to operator
  for (int i = start + 1; i < end; ++i)
  {
    auto *cur_op = operators.at(i);
    assert(cur_op != nullptr);

    runtime_graph->addOperatorStatus(cur_op, OperationGraphStatus::MIDDLE);
  }
}

// TODO: support more operations
bool supportedOperationsForIntermediateQuantization(int operation_index, CircleReader *reader)
{
  const auto operators = reader->operators();
  const auto cur_operation = operators.at(operation_index);

  const auto code = reader->builtin_code(cur_operation);

  if (isInplaceOperation(code))
    return true;

  switch (code)
  {
    case circle::BuiltinOperator_FULLY_CONNECTED:
    case circle::BuiltinOperator_CONV_2D:
    case circle::BuiltinOperator_STRIDED_SLICE:
    case circle::BuiltinOperator_MAX_POOL_2D:
    case circle::BuiltinOperator_MUL:
    case circle::BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM:
      return true;
    default:
      break;
  }
  return false;
}

} // namespace

void GraphLoader::checkIntermediateQuantization(CircleReader *reader, RuntimeGraph *runtime_graph)
{
  // Vector of pairs: memory usage, index
  // memory usage - needed memory for intermediate tensors to run operation at position = index
  // index - position of the operation in graph
  std::vector<int> operations_mem_usage;
  buildMemoryUsageVector(reader, operations_mem_usage);
  auto max_element = std::max_element(operations_mem_usage.begin(), operations_mem_usage.end());

  if (max_element == operations_mem_usage.end())
    return;

  int target = *std::max_element(operations_mem_usage.begin(), operations_mem_usage.end()) / 2;
  int start = 0;
  int end = 0;
  std::vector<std::pair<int, int>> ranges;
  for (int i = 0; i < operations_mem_usage.size(); ++i)
  {
    if (operations_mem_usage[i] >= target and
        supportedOperationsForIntermediateQuantization(i, reader))
    {
      end++;
    }
    else
    {
      if (start != end)
        ranges.push_back({start, end});

      start = end = i;
    }
  }
  if (start != end)
    ranges.push_back({start, end});

  for (const auto &range : ranges)
  {
    addOperationsStatusForRange(range.first, range.second, reader, runtime_graph);
  }

  return;
}
void GraphLoader::checkInplaceOps(CircleReader *reader, RuntimeGraph *runtime_graph)
{
  const auto operators = reader->operators();
  const auto graph_outputs = reader->outputs();
  for (uint32_t i = 0; i < operators.size(); ++i)
  {
    const auto *op = operators.at(i);
    assert(op != nullptr);

    // Check inplace optimization for operation with single input and single output
    if (isInplaceOperation(reader->builtin_code(op)))
    {
      const auto *op_inputs = op->inputs();
      const auto *op_outputs = op->outputs();

      bool is_inplace = true;
      auto non_const_input_it = op_inputs->begin();
      while (true)
      {
        non_const_input_it =
          std::find_if(non_const_input_it, op_inputs->end(), [&reader](const auto input_idx) {
            if (input_idx == -1)
              return false;

            return not Tensor::is_constant_tensor(reader, reader->tensors()[input_idx]);
          });

        if (non_const_input_it == op_inputs->end())
          break;

        auto dist = std::distance(op_inputs->begin(), non_const_input_it);

        const auto non_const_input_idx = *non_const_input_it;

        // Check single usage of input tensor
        if (not isSingleUsageOfTensor(reader, non_const_input_idx))
        {
          is_inplace = false;
          break;
        }

        // Let's check single usage of output tensor
        if (dist >= op_outputs->size() and op_outputs->size() == 1)
          dist = 0;
        assert(dist < op_outputs->size());
        const auto output_index = op_outputs->operator[](dist);
        if (not isSingleUsageOfTensor(reader, output_index))
        {
          is_inplace = false;
          break;
        }

        // Check that num elements are equal
        {
          const auto *input_non_const_tensor = reader->tensors().at(non_const_input_idx);
          const auto *output_tensor = reader->tensors().at(output_index);
          if (Tensor::num_elements(input_non_const_tensor) != Tensor::num_elements(output_tensor))
          {
            is_inplace = false;
            break;
          }
        }

        // Let's check that output is not a graph output tensor
        {
          if (std::find(graph_outputs.begin(), graph_outputs.end(), output_index) !=
              graph_outputs.end())
          {
            is_inplace = false;
            break;
          }
        }

        non_const_input_it++;
      }

      if (is_inplace)
        runtime_graph->addInplaceOpIndex(op);
    }
  }
}

} // namespace luci_interpreter
