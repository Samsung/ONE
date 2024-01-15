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

#include <mio/tflite/schema_generated.h>

#include <map>
#include <string>
#include <vector>

namespace tflinspect
{

template <typename T> std::vector<T> as_index_vector(const flatbuffers::Vector<T> *flat_array)
{
  std::vector<T> ret(flat_array->size());
  for (uint32_t i = 0; i < flat_array->size(); i++)
  {
    ret[i] = flat_array->Get(i);
  }
  return ret;
}

/**
 * @brief Loads TF lite file and provides helpers to access attributes
 */
class Reader
{
private:
  using TFliteSubGraphs_t = flatbuffers::Vector<flatbuffers::Offset<tflite::SubGraph>>;
  using TFliteBuffers_t = flatbuffers::Vector<flatbuffers::Offset<tflite::Buffer>>;
  using TFliteTensors_t = flatbuffers::Vector<flatbuffers::Offset<tflite::Tensor>>;
  using TFliteOperators_t = flatbuffers::Vector<flatbuffers::Offset<tflite::Operator>>;

public:
  Reader(const tflite::Model *model);

  Reader() = delete;

public:
  const std::vector<const tflite::OperatorCode *> &opcodes() { return _op_codes; }
  const TFliteBuffers_t *buffers() { return _buffers; }
  const TFliteTensors_t *tensors() { return _tensors; }
  const TFliteOperators_t *operators() { return _operators; }
  const std::vector<int32_t> &inputs() const { return _inputs; }
  const std::vector<int32_t> &outputs() const { return _outputs; }

  uint32_t num_subgraph() const { return _subgraphs->size(); }

  size_t buffer_info(uint32_t buf_idx, const uint8_t **buff_data);
  tflite::BuiltinOperator builtin_code(const tflite::Operator *op) const;
  std::string opcode_name(const tflite::Operator *op) const;

public:
  bool select_subgraph(uint32_t subgraph);

private:
  const TFliteSubGraphs_t *_subgraphs{nullptr};
  const TFliteBuffers_t *_buffers{nullptr};
  const TFliteTensors_t *_tensors{nullptr};
  const TFliteOperators_t *_operators{nullptr};

  std::vector<const tflite::OperatorCode *> _op_codes;
  std::vector<int32_t> _inputs;
  std::vector<int32_t> _outputs;
};

} // namespace tflinspect

#endif // __READER_H__
