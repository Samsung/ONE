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

#ifndef __CIRCLEREAD_READ_H__
#define __CIRCLEREAD_READ_H__

#include <mio/circle/schema_generated.h>

#include <map>
#include <string>
#include <vector>

namespace circleread
{

template <typename T> std::vector<T> as_index_vector(const flatbuffers::Vector<T> *flat_array)
{
  if (flat_array == nullptr)
  {
    throw std::runtime_error("flat array is nullptr");
  }

  std::vector<T> ret(flat_array->Length());
  for (uint32_t i = 0; i < flat_array->Length(); i++)
  {
    ret[i] = flat_array->Get(i);
  }
  return ret;
}

bool is_valid(const circle::OperatorCode *opcode);
bool is_custom(const circle::OperatorCode *opcode);
std::string opcode_name(const circle::OperatorCode *opcode);
const char *tensor_type(const circle::Tensor *tensor);
const char *tensor_name(const circle::Tensor *tensor);

/**
 * @brief Loads Circle file and provides helpers to access attributes
 */
class Reader
{
private:
  using CircleSubGraphs_t = flatbuffers::Vector<flatbuffers::Offset<circle::SubGraph>>;
  using CircleBuffers_t = flatbuffers::Vector<flatbuffers::Offset<circle::Buffer>>;
  using CircleTensors_t = flatbuffers::Vector<flatbuffers::Offset<circle::Tensor>>;
  using CircleOperators_t = flatbuffers::Vector<flatbuffers::Offset<circle::Operator>>;
  using CircleMetadata_t = flatbuffers::Vector<flatbuffers::Offset<circle::Metadata>>;

public:
  Reader(const circle::Model *model);

  Reader() = delete;

public:
  uint32_t version() const { return _version; }

  const std::vector<const circle::OperatorCode *> &opcodes() { return _op_codes; }
  const CircleBuffers_t *buffers() { return _buffers; }
  const CircleTensors_t *tensors() { return _tensors; }
  const CircleOperators_t *operators() { return _operators; }
  const std::vector<int32_t> &inputs() const { return _inputs; }
  const std::vector<int32_t> &outputs() const { return _outputs; }
  const circle::DataFormat &data_format() const { return _data_format; }
  const CircleMetadata_t *metadata() const { return _metadata; }

  uint32_t num_subgraph() const { return _subgraphs->Length(); }

  size_t buffer_info(uint32_t buf_idx, const uint8_t **buff_data);
  circle::BuiltinOperator builtin_code(const circle::Operator *op) const;
  std::string opcode_name(const circle::Operator *op) const;

public:
  bool select_subgraph(uint32_t subgraph);
  const std::string &subgraph_name(void) const { return _subgraph_name; }
  uint32_t subgraph_index(void) const { return _subgraph_index; }

private:
  uint32_t _version;

  const CircleSubGraphs_t *_subgraphs{nullptr};
  const CircleBuffers_t *_buffers{nullptr};
  const CircleTensors_t *_tensors{nullptr};
  const CircleOperators_t *_operators{nullptr};
  const CircleMetadata_t *_metadata{nullptr};

  uint32_t _subgraph_index = 0;
  std::string _subgraph_name;
  std::vector<const circle::OperatorCode *> _op_codes;
  std::vector<int32_t> _inputs;
  std::vector<int32_t> _outputs;
  circle::DataFormat _data_format = circle::DataFormat::DataFormat_CHANNELS_FIRST;
};

} // namespace circleread

#endif // __CIRCLEREAD_READ_H__
