/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "TFliteImport.h"

#include "Convert.h"

#include <mio_tflite2121/Helper.h>

#include <sstream>

namespace tflchef
{

TFliteImport::TFliteImport(const tflite::Model *model)
{
  _subgraphs = model->subgraphs();
  _buffers = model->buffers();

  auto opcodes = model->operator_codes();
  for (const ::tflite::OperatorCode *opcode : *opcodes)
  {
    _op_codes.push_back(opcode);
  }
}

bool TFliteImport::select_sub_graph(uint32_t sgindex)
{
  _tensors = nullptr;
  _operators = nullptr;
  _inputs.clear();
  _outputs.clear();

  if (_subgraphs->size() <= sgindex)
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

tflite::BuiltinOperator TFliteImport::builtin_code(const tflite::Operator *op) const
{
  uint32_t index = op->opcode_index();
  assert(index < _op_codes.size());
  const tflite::OperatorCode *opcode = _op_codes.at(index);

  return mio::tflite::builtin_code_neutral(opcode);
}

std::string TFliteImport::opcode_name(const tflite::Operator *op) const
{
  uint32_t index = op->opcode_index();
  assert(index < _op_codes.size());
  const tflite::OperatorCode *opcode = _op_codes.at(index);

  if (!mio::tflite::is_valid(opcode))
  {
    std::ostringstream oss;
    oss << "(invalid: " << index << ")";
    return oss.str();
  }

  if (mio::tflite::is_custom(opcode))
  {
    if (!opcode->custom_code())
      return "(invalid custom)";

    return opcode->custom_code()->c_str();
  }

  tflite::BuiltinOperator code = mio::tflite::builtin_code_neutral(opcode);
  return EnumNameBuiltinOperator(code);
}

size_t TFliteImport::buffer_info(const tflite::Tensor *tensor, const uint8_t **buff_data)
{
  *buff_data = nullptr;

  if (tensor->buffer() == 0)
    return 0;

  if (auto *buffer = (*_buffers)[tensor->buffer()])
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

} // namespace tflchef
