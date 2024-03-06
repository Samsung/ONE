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

#include "core/reader/OMCircleReader.h"

using namespace onert_micro::core::reader;
using namespace onert_micro;

namespace
{

template <typename VECTORTYPE> uint32_t read_u32(const VECTORTYPE &buffer, uint32_t idx)
{
  uint32_t val = 0;
  val += (buffer->operator[](idx + 0) << 0 * 8);
  val += (buffer->operator[](idx + 1) << 1 * 8);
  val += (buffer->operator[](idx + 2) << 2 * 8);
  val += (buffer->operator[](idx + 3) << 3 * 8);
  return val;
}

// 'map_tensors_indexes' is decoded to std::map<uint32_t, uint32_t> format.
template <typename VECTORTYPE>
const std::unordered_map<uint16_t, uint16_t> decoded_map_tensors_indexes(const VECTORTYPE &map_tensors_indexes_data)
{
  std::unordered_map<uint16_t, uint16_t> map_tensors_indexes;
  uint32_t idx = 0;

  assert(map_tensors_indexes_data->size() >= 4 && "Op table decode error : invalid entry number");

  uint32_t entry_number = read_u32(map_tensors_indexes_data, idx);
  idx += sizeof(uint32_t);

  while (idx < map_tensors_indexes_data->size())
  {
    assert(idx + 2 * sizeof(uint32_t) <= map_tensors_indexes_data->size() && "Op table decode error : invalid entry item");

    uint32_t id = read_u32(map_tensors_indexes_data, idx);
    idx += sizeof(uint32_t);

    uint32_t origin_id = read_u32(map_tensors_indexes_data, idx);
    idx += sizeof(uint32_t);

    assert(map_tensors_indexes.insert({id, origin_id}).second != false && "Op table decode error : duplicated origin ID");
  }

  assert(idx == map_tensors_indexes_data->size() && "Op table decode error : data size invalid");

  assert(map_tensors_indexes.size() == entry_number && "Op table decode error : entry number invalid");

  return map_tensors_indexes;
}

circle::BuiltinOperator builtin_code_neutral(const ::circle::OperatorCode *opcode)
{
  assert(opcode != nullptr);
  if (opcode->deprecated_builtin_code() == 127)
  {
    assert(opcode->builtin_code() >= 127);
    return opcode->builtin_code();
  }
  // There was no 255(-1) value in v0.3
  assert(opcode->deprecated_builtin_code() != -1);
  return static_cast<::circle::BuiltinOperator>(opcode->deprecated_builtin_code());
}

} // namespace

circle::BuiltinOperator OMCircleReader::builtin_code(const circle::Operator *op) const
{
  assert(op != nullptr);

  const auto op_codes = opcodes();
  uint32_t index = op->opcode_index();

  assert(index < op_codes->size());

  const auto opcode = op_codes->operator[](index);
  assert(opcode != nullptr);

  return builtin_code_neutral(opcode);
}

OMStatus OMCircleReader::parse(const char *model_ptr)
{
  assert(_model == nullptr && "Already init _model");
  assert(model_ptr != nullptr && "Model pointer cannot be null");
  if (model_ptr == nullptr or _model != nullptr)
  {
    return UnknownError;
  }

  _model = circle::GetModel(model_ptr);

  return Ok;
}

OMStatus OMCircleReader::select_subgraph(uint32_t sgindex)
{
  if (num_subgraph() <= sgindex)
    return UnknownError;

  auto subgraphs = _model->subgraphs();
  if (subgraphs == nullptr)
    return UnknownError;

  _current_subgraph = subgraphs->Get(sgindex);
  if (_current_subgraph == nullptr)
    return UnknownError;

  _current_subgraph_index = sgindex;

  return Ok;
}

bool OMCircleReader::isConstTensor(uint32_t tensor_index)
{
  if (tensor_index == -1)
    return false;

  const auto tmp_tensor = _current_subgraph->tensors()->operator[](tensor_index);
  return _model->buffers()->operator[](tmp_tensor->buffer())->data() != nullptr;
}

std::unordered_map<uint16_t, uint16_t> OMCircleReader::readTensorsTrainIndexesTable()
{
  const auto *metadata_ptr = metadata();
  const auto *buffers_ptr = buffers();
  for (uint32_t i = 0; i < metadata_ptr->size(); ++i)
  {
    const circle::Metadata *meta = metadata_ptr->operator[](i);
    assert(meta != nullptr);

    assert(meta->buffer() < buffers_ptr->size());
    assert(buffers_ptr->operator[](meta->buffer()) != nullptr);
    const auto buffer = buffers_ptr->operator[](meta->buffer())->data();

    assert(meta->name() != nullptr);
    if (meta->name()->str().compare("ONE_train_inform") == 0)
      return decoded_map_tensors_indexes(buffer);
  }

  return {};
}
