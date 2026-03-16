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

#include "Context.h"

#include "Convert.h"

#include <coco/IR/Data.h>
#include <coco/IR/Module.h>

#include <nncc/core/ADT/tensor/Shape.h>
#include <schema_generated.h>

#include <map>
#include <sstream>

using namespace nncc::core::ADT;

namespace tflimport
{

void TensorContext::prepare(const tflite::SubGraph *graph)
{
  for (uint32_t tensor_id = 0; tensor_id < graph->tensors()->size(); ++tensor_id)
  {
    auto const tensor_info = graph->tensors()->Get(tensor_id);
    auto const tensor_name = tensor_info->name()->str();
    auto const tensor_shape = as_tensor_shape(tensor_info->shape());
    auto const tensor_type = tensor_info->type();

    _name_ctx[tensor_id] = tensor_name;
    _shape_ctx[tensor_id] = tensor_shape;
    _type_ctx[tensor_id] = tensor_type;
  }
}

TflOpCodeContext::TflOpCodeContext(
  const flatbuffers::Vector<flatbuffers::Offset<tflite::OperatorCode>> *opcodes)
{
  for (const tflite::OperatorCode *opcode : *opcodes)
  {
    _opcodes.push_back(opcode);
  }
}

tflite::BuiltinOperator TflOpCodeContext::builtin_code(const tflite::Operator *op) const
{
  uint32_t index = op->opcode_index();
  assert(index < _opcodes.size());
  const tflite::OperatorCode *opcode = _opcodes.at(index);
  return opcode->builtin_code();
}

std::string TflOpCodeContext::opcode_name(const tflite::Operator *op) const
{
  uint32_t index = op->opcode_index();
  assert(index < _opcodes.size());
  const tflite::OperatorCode *opcode = _opcodes.at(index);

  if (!is_valid(opcode))
  {
    std::ostringstream oss;
    oss << "(invalid: " << index << ")";
    return oss.str();
  }

  if (is_custom(opcode))
  {
    if (!opcode->custom_code())
      return "(invalid custom)";

    return opcode->custom_code()->c_str();
  }

  tflite::BuiltinOperator code = opcode->builtin_code();
  return EnumNameBuiltinOperator(code);
}

bool TflOpCodeContext::is_valid(const tflite::OperatorCode *opcode)
{
  tflite::BuiltinOperator code = opcode->builtin_code();
  return (tflite::BuiltinOperator_MIN <= code && code <= tflite::BuiltinOperator_MAX);
}

bool TflOpCodeContext::is_custom(const tflite::OperatorCode *opcode)
{
  tflite::BuiltinOperator code = opcode->builtin_code();
  return (code == tflite::BuiltinOperator_CUSTOM);
}

TflBufferContext::TflBufferContext(const tflite::Model *tfl_model)
{
  const flatbuffers::Vector<flatbuffers::Offset<tflite::Buffer>> *tfl_buffers;

  tfl_buffers = tfl_model->buffers();

  for (uint32_t buffer_id = 0; buffer_id < tfl_buffers->size(); ++buffer_id)
  {
    _buffer_ctx[buffer_id] = (*tfl_buffers)[buffer_id];
  }
}

} // namespace tflimport
