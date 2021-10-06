/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Reader.h"

#include <cassert>
#include <sstream>
#include <string>

namespace tflinspect
{

// This will provide v3/v3a format neutral BuiltinOperator
tflite::BuiltinOperator builtin_code_neutral(const tflite::OperatorCode *opcode)
{
  assert(opcode != nullptr);
  // TODO support v3a
  return opcode->builtin_code();
}

bool is_valid(const tflite::OperatorCode *opcode)
{
  tflite::BuiltinOperator code = builtin_code_neutral(opcode);
  return (tflite::BuiltinOperator_MIN <= code && code <= tflite::BuiltinOperator_MAX);
}

bool is_custom(const tflite::OperatorCode *opcode)
{
  tflite::BuiltinOperator code = builtin_code_neutral(opcode);
  return (code == tflite::BuiltinOperator_CUSTOM);
}

std::string opcode_name(const tflite::OperatorCode *opcode)
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

  tflite::BuiltinOperator code = builtin_code_neutral(opcode);
  return tflite::EnumNameBuiltinOperator(code);
}

const char *tensor_type(const tflite::Tensor *tensor)
{
  return tflite::EnumNameTensorType(tensor->type());
}

const char *tensor_name(const tflite::Tensor *tensor)
{
  static const char *kEmptyTensorName = "(noname)";

  auto name = tensor->name();
  if (name)
    return name->c_str();

  return kEmptyTensorName;
}

Reader::Reader(const tflite::Model *model)
{
  _subgraphs = model->subgraphs();
  _buffers = model->buffers();

  auto opcodes = model->operator_codes();
  for (const ::tflite::OperatorCode *opcode : *opcodes)
  {
    _op_codes.push_back(opcode);
  }
}

size_t Reader::buffer_info(uint32_t buf_idx, const uint8_t **buff_data)
{
  if (buff_data != nullptr)
  {
    *buff_data = nullptr;
  }

  if (buf_idx == 0)
    return 0;

  if (auto *buffer = (*_buffers)[buf_idx])
  {
    if (auto *array = buffer->data())
    {
      if (size_t size = array->size())
      {
        if (buff_data != nullptr)
        {
          *buff_data = reinterpret_cast<const uint8_t *>(array->data());
        }
        return size;
      }
    }
  }

  return 0;
}

tflite::BuiltinOperator Reader::builtin_code(const tflite::Operator *op) const
{
  uint32_t index = op->opcode_index();
  assert(index < _op_codes.size());
  const tflite::OperatorCode *opcode = _op_codes.at(index);

  return tflinspect::builtin_code_neutral(opcode);
}

std::string Reader::opcode_name(const tflite::Operator *op) const
{
  uint32_t index = op->opcode_index();
  assert(index < _op_codes.size());
  const tflite::OperatorCode *opcode = _op_codes.at(index);

  if (!is_valid(opcode))
  {
    std::ostringstream oss;
    oss << "(invalid: " << index << ")";
    return oss.str();
  }

  return tflinspect::opcode_name(opcode);
}

bool Reader::select_subgraph(uint32_t sgindex)
{
  _tensors = nullptr;
  _operators = nullptr;

  _inputs.clear();
  _outputs.clear();

  if (_subgraphs->Length() <= sgindex)
  {
    assert(false);
    return false;
  }

  const tflite::SubGraph *subgraph = (*_subgraphs)[sgindex];

  _tensors = subgraph->tensors();
  _operators = subgraph->operators();

  _inputs = as_index_vector(subgraph->inputs());
  _outputs = as_index_vector(subgraph->outputs());

  return true;
}

} // namespace tflinspect
