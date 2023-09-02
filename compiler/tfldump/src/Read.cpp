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

#include "Read.h"

#include <mio_tflite2121/Helper.h>

#include <sstream>
#include <string>

namespace tflread
{

Reader::Reader(const tflite::Model *model)
{
  _version = model->version();
  _subgraphs = model->subgraphs();
  _buffers = model->buffers();
  _metadata = model->metadata();
  _signaturedefs = model->signature_defs();

  auto opcodes = model->operator_codes();
  for (const ::tflite::OperatorCode *opcode : *opcodes)
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

tflite::BuiltinOperator Reader::builtin_code(const tflite::Operator *op) const
{
  uint32_t index = op->opcode_index();
  assert(index < _op_codes.size());
  const tflite::OperatorCode *opcode = _op_codes.at(index);

  return mio::tflite::builtin_code_neutral(opcode);
}

std::string Reader::opcode_name(const tflite::Operator *op) const
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

  return mio::tflite::opcode_name(opcode);
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

  const tflite::SubGraph *subgraph = (*_subgraphs)[sgindex];

  auto name = subgraph->name();
  _subgraph_name = name ? name->c_str() : "(noname)";

  _tensors = subgraph->tensors();
  _operators = subgraph->operators();

  _inputs = as_index_vector(subgraph->inputs());
  _outputs = as_index_vector(subgraph->outputs());

  return true;
}

} // namespace tflread
