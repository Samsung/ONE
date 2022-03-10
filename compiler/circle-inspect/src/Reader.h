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

#ifndef __READER_H__
#define __READER_H__

#include <mio/circle/schema_generated.h>

#include <map>
#include <string>
#include <vector>

namespace circleinspect
{

template <typename T> std::vector<T> as_index_vector(const flatbuffers::Vector<T> *flat_array)
{
  std::vector<T> ret(flat_array->Length());
  for (uint32_t i = 0; i < flat_array->Length(); i++)
  {
    ret[i] = flat_array->Get(i);
  }
  return ret;
}

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

public:
  Reader(const circle::Model *model);

  Reader() = delete;

public:
  const std::vector<const circle::OperatorCode *> &opcodes() { return _op_codes; }
  const CircleBuffers_t *buffers() { return _buffers; }
  const CircleTensors_t *tensors() { return _tensors; }
  const CircleOperators_t *operators() { return _operators; }
  const std::vector<int32_t> &inputs() const { return _inputs; }
  const std::vector<int32_t> &outputs() const { return _outputs; }

  uint32_t num_subgraph() const { return _subgraphs->Length(); }

  size_t buffer_info(uint32_t buf_idx, const uint8_t **buff_data);
  circle::BuiltinOperator builtin_code(const circle::Operator *op) const;
  std::string opcode_name(const circle::Operator *op) const;
  std::string tensor_name(const circle::Tensor *tensor) const;
  std::string tensor_dtype(const circle::Tensor *tensor) const;

public:
  bool select_subgraph(uint32_t subgraph);

private:
  const CircleSubGraphs_t *_subgraphs{nullptr};
  const CircleBuffers_t *_buffers{nullptr};
  const CircleTensors_t *_tensors{nullptr};
  const CircleOperators_t *_operators{nullptr};

  std::vector<const circle::OperatorCode *> _op_codes;
  std::vector<int32_t> _inputs;
  std::vector<int32_t> _outputs;
};

} // namespace circleinspect

#endif // __READER_H__
