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

#ifndef __CIRCLE_IMPORT_H__
#define __CIRCLE_IMPORT_H__

#include <mio/circle/schema_generated.h>

#include <souschef/TensorFiller.h>

#include <circlechef.pb.h>

#include <map>
#include <vector>

namespace circlechef
{

using CircleSubGraphs_t = flatbuffers::Vector<flatbuffers::Offset<circle::SubGraph>>;
using CircleTensors_t = flatbuffers::Vector<flatbuffers::Offset<circle::Tensor>>;
using CircleBuffers_t = flatbuffers::Vector<flatbuffers::Offset<circle::Buffer>>;
using CircleOperators_t = flatbuffers::Vector<flatbuffers::Offset<circle::Operator>>;

/**
 * @brief Loads TF lite file and provides helpers to access attributes
 */
class CircleImport : public souschef::TensorFiller
{
public:
  CircleImport(const circle::Model *model);

  CircleImport() = delete;

public:
  bool select_sub_graph(uint32_t subgraph);

public:
  const CircleBuffers_t *buffers() { return _buffers; }
  const CircleTensors_t *tensors() { return _tensors; }
  const CircleOperators_t *operators() { return _operators; }
  const std::vector<int32_t> &inputs() const { return _inputs; }
  const std::vector<int32_t> &outputs() const { return _outputs; }

  uint32_t num_subgraph() const { return _subgraphs->size(); }

  circle::BuiltinOperator builtin_code(const circle::Operator *op) const;
  std::string opcode_name(const circle::Operator *op) const;
  size_t buffer_info(const circle::Tensor *tensor, const uint8_t **buff_data);

private:
  const CircleSubGraphs_t *_subgraphs{nullptr};
  const CircleBuffers_t *_buffers{nullptr};
  const CircleTensors_t *_tensors{nullptr};
  const CircleOperators_t *_operators{nullptr};

  std::vector<const circle::OperatorCode *> _op_codes{};
  std::vector<int32_t> _inputs{};
  std::vector<int32_t> _outputs{};
};

} // namespace circlechef

#endif // __CIRCLE_IMPORT_H__
