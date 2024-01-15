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

#ifndef __TFLITE_IMPORT_H__
#define __TFLITE_IMPORT_H__

#include <mio/tflite/schema_generated.h>

#include <souschef/TensorFiller.h>

#include <tflchef.pb.h>

#include <map>
#include <vector>

namespace tflchef
{

using TFliteSubGraphs_t = flatbuffers::Vector<flatbuffers::Offset<tflite::SubGraph>>;
using TFliteTensors_t = flatbuffers::Vector<flatbuffers::Offset<tflite::Tensor>>;
using TFliteBuffers_t = flatbuffers::Vector<flatbuffers::Offset<tflite::Buffer>>;
using TFliteOperators_t = flatbuffers::Vector<flatbuffers::Offset<tflite::Operator>>;

/**
 * @brief Loads TF lite file and provides helpers to access attributes
 */
class TFliteImport : public souschef::TensorFiller
{
public:
  TFliteImport(const tflite::Model *model);

  TFliteImport() = delete;

public:
  bool select_sub_graph(uint32_t subgraph);

public:
  const TFliteBuffers_t *buffers() { return _buffers; }
  const TFliteTensors_t *tensors() { return _tensors; }
  const TFliteOperators_t *operators() { return _operators; }
  const std::vector<int32_t> &inputs() const { return _inputs; }
  const std::vector<int32_t> &outputs() const { return _outputs; }

  uint32_t num_subgraph() const { return _subgraphs->size(); }

  tflite::BuiltinOperator builtin_code(const tflite::Operator *op) const;
  std::string opcode_name(const tflite::Operator *op) const;
  size_t buffer_info(const tflite::Tensor *tensor, const uint8_t **buff_data);

private:
  const TFliteSubGraphs_t *_subgraphs{nullptr};
  const TFliteBuffers_t *_buffers{nullptr};
  const TFliteTensors_t *_tensors{nullptr};
  const TFliteOperators_t *_operators{nullptr};

  std::vector<const tflite::OperatorCode *> _op_codes{};
  std::vector<int32_t> _inputs{};
  std::vector<int32_t> _outputs{};
};

} // namespace tflchef

#endif // __TFLITE_IMPORT_H__
