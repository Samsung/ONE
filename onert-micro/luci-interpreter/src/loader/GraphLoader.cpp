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

// TODO: add more operations
bool isCouldBeEmplaceOperation(circle::BuiltinOperator op)
{
  switch (op)
  {
    case circle::BuiltinOperator_LOGISTIC:
    case circle::BuiltinOperator_RESHAPE:
    case circle::BuiltinOperator_EXPAND_DIMS:
      return true;
    default:
      return false;
  }
}

bool isCouldBeEmplaceTensor(CircleReader *reader, const int32_t tensor_index)
{
  uint32_t usage_count = 0;
  for (uint32_t i = 0; i < reader->operators().size(); ++i)
  {
    const auto op = reader->operators().at(i);
    assert(op != nullptr);

    for (int32_t j = 0; j < op->inputs()->size(); ++j)
    {
      const auto input_index = op->inputs()->operator[](j);
      if (input_index == tensor_index)
        usage_count++;

      if (usage_count > 1)
        return false;
    }
  }
  return true;
}

} // namespace

void GraphLoader::checkInplaceOps(CircleReader *reader, RuntimeGraph *runtime_graph)
{
  for (uint32_t i = 0; i < reader->operators().size(); ++i)
  {
    const auto *op = reader->operators().at(i);
    assert(op != nullptr);

    bool is_graph_input = false;
    for (int32_t j = 0; j < op->inputs()->size(); ++j)
    {
      const auto input_index = op->inputs()->operator[](j);
      if (input_index == -1)
        continue;

      const auto &inputs_indexes = reader->inputs();

      is_graph_input = (std::find(inputs_indexes.begin(), inputs_indexes.end(), input_index) !=
                        inputs_indexes.end()) or
                       is_graph_input;

      if (not is_graph_input and isCouldBeEmplaceOperation(reader->builtin_code(op)) and
          op->outputs()->size() == 1 and isCouldBeEmplaceTensor(reader, input_index))
      {
        runtime_graph->addInplaceOpIndex(i);
      }
    }
  }
}

} // namespace luci_interpreter
