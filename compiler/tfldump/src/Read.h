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

#ifndef __TFLREAD_READ_H__
#define __TFLREAD_READ_H__

#include <mio/tflite/schema_generated.h>

#include <map>
#include <string>
#include <vector>

namespace tflread
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
  using TFliteMetadata_t = flatbuffers::Vector<flatbuffers::Offset<tflite::Metadata>>;
  using TFliteSignatureDef_t = flatbuffers::Vector<flatbuffers::Offset<tflite::SignatureDef>>;

public:
  Reader(const tflite::Model *model);

  Reader() = delete;

public:
  uint32_t version() const { return _version; }

  const std::vector<const tflite::OperatorCode *> &opcodes() { return _op_codes; }
  const TFliteBuffers_t *buffers() { return _buffers; }
  const TFliteTensors_t *tensors() { return _tensors; }
  const TFliteOperators_t *operators() { return _operators; }
  const std::vector<int32_t> &inputs() const { return _inputs; }
  const std::vector<int32_t> &outputs() const { return _outputs; }
  const TFliteMetadata_t *metadata() const { return _metadata; }
  const TFliteSignatureDef_t *signaturedefs() const { return _signaturedefs; }

  uint32_t num_subgraph() const { return _subgraphs->size(); }

  size_t buffer_info(uint32_t buf_idx, const uint8_t **buff_data);
  tflite::BuiltinOperator builtin_code(const tflite::Operator *op) const;
  std::string opcode_name(const tflite::Operator *op) const;

public:
  bool select_subgraph(uint32_t subgraph);
  const std::string &subgraph_name(void) const { return _subgraph_name; }
  uint32_t subgraph_index(void) const { return _subgraph_index; }

private:
  uint32_t _version;

  const TFliteSubGraphs_t *_subgraphs{nullptr};
  const TFliteBuffers_t *_buffers{nullptr};
  const TFliteTensors_t *_tensors{nullptr};
  const TFliteOperators_t *_operators{nullptr};
  const TFliteMetadata_t *_metadata{nullptr};
  const TFliteSignatureDef_t *_signaturedefs{nullptr};

  uint32_t _subgraph_index;
  std::string _subgraph_name;
  std::vector<const tflite::OperatorCode *> _op_codes;
  std::vector<int32_t> _inputs;
  std::vector<int32_t> _outputs;
};

} // namespace tflread

#endif // __TFLREAD_READ_H__
