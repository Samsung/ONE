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

#include "Read.h"

#include <sstream>
#include <string>

namespace circleread
{

bool is_valid(const circle::OperatorCode *opcode)
{
  circle::BuiltinOperator code = opcode->builtin_code();
  return (circle::BuiltinOperator_MIN <= code && code <= circle::BuiltinOperator_MAX);
}

bool is_custom(const circle::OperatorCode *opcode)
{
  circle::BuiltinOperator code = opcode->builtin_code();
  return (code == circle::BuiltinOperator_CUSTOM);
}

std::string opcode_name(const circle::OperatorCode *opcode)
{
  assert(opcode);

  if (!is_valid(opcode))
  {
    std::ostringstream oss;
    oss << "(invalid)";
    return oss.str();
  }

  if (is_custom(opcode))
  {
    if (!opcode->custom_code())
      return "(invalid custom)";

    std::string custom_op = "CUSTOM(";
    custom_op += opcode->custom_code()->c_str();
    custom_op += ")";
    return custom_op;
  }

  circle::BuiltinOperator code = opcode->builtin_code();
  return circle::EnumNameBuiltinOperator(code);
}

const char *tensor_type(const circle::Tensor *tensor)
{
  return circle::EnumNameTensorType(tensor->type());
}

const char *tensor_name(const circle::Tensor *tensor)
{
  static const char *kEmptyTensorName = "(noname)";

  auto name = tensor->name();
  if (name)
    return name->c_str();

  return kEmptyTensorName;
}

Reader::Reader(const circle::Model *model)
{
  _version = model->version();
  _subgraphs = model->subgraphs();
  _buffers = model->buffers();
  _metadata = model->metadata();

  auto opcodes = model->operator_codes();
  for (const ::circle::OperatorCode *opcode : *opcodes)
  {
    _op_codes.push_back(opcode);
  }
}

size_t Reader::buffer_info(uint32_t buf_idx, const uint8_t **buff_data)
{
  *buff_data = nullptr;

  if (buf_idx == 0)
    return 0;

  if (auto *buffer = (*_buffers)[buf_idx])
  {
    if (auto *array = buffer->data())
    {
      if (size_t size = array->size())
      {
        *buff_data = reinterpret_cast<const uint8_t *>(array->data());
        return size;
      }
    }
  }

  return 0;
}

circle::BuiltinOperator Reader::builtin_code(const circle::Operator *op) const
{
  uint32_t index = op->opcode_index();
  assert(index < _op_codes.size());
  const circle::OperatorCode *opcode = _op_codes.at(index);

  return opcode->builtin_code();
}

std::string Reader::opcode_name(const circle::Operator *op) const
{
  uint32_t index = op->opcode_index();
  assert(index < _op_codes.size());
  const circle::OperatorCode *opcode = _op_codes.at(index);

  if (!is_valid(opcode))
  {
    std::ostringstream oss;
    oss << "(invalid: " << index << ")";
    return oss.str();
  }

  return circleread::opcode_name(opcode);
}

bool Reader::select_subgraph(uint32_t sgindex)
{
  _subgraph_index = sgindex;
  _tensors = nullptr;
  _operators = nullptr;

  _inputs.clear();
  _outputs.clear();

  if (_subgraphs->Length() <= sgindex)
  {
    assert(false);
    return false;
  }

  const circle::SubGraph *subgraph = (*_subgraphs)[sgindex];

  auto name = subgraph->name();
  _subgraph_name = name ? name->c_str() : "(noname)";

  _tensors = subgraph->tensors();
  _operators = subgraph->operators();
  _data_format = subgraph->data_format();

  _inputs = as_index_vector(subgraph->inputs());
  _outputs = as_index_vector(subgraph->outputs());

  return true;
}

} // namespace circleread
