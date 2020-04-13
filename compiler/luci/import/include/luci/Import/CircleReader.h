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

#ifndef __LUCI_IMPORT_GRAPHREADER_H__
#define __LUCI_IMPORT_GRAPHREADER_H__

#include <mio/circle/schema_generated.h>

#include <luci/IR/AttrFusedActFunc.h>
#include <luci/IR/AttrPadding.h>
#include <luci/IR/CircleQuantParam.h>

#include <loco.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace luci
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

bool is_valid(const circle::OperatorCode *opcode);
bool is_custom(const circle::OperatorCode *opcode);
std::string opcode_name(const circle::OperatorCode *opcode);
const char *tensor_type(const circle::Tensor *tensor);
const char *tensor_name(const circle::Tensor *tensor);
const circle::QuantizationParameters *tensor_quantization(const circle::Tensor *tensor);

loco::DataType luci_datatype(circle::TensorType type);
loco::DataType luci_datatype(const circle::Tensor *tensor);
FusedActFunc luci_actfunc(const circle::ActivationFunctionType type);
Padding luci_padding(const circle::Padding padding);
std::unique_ptr<CircleQuantParam>
luci_quantparam(const circle::QuantizationParameters *quantization);

/**
 * @brief Loads Circle file and provides helpers to access attributes
 */
class CircleReader
{
private:
  using CircleSubGraphs_t = flatbuffers::Vector<flatbuffers::Offset<circle::SubGraph>>;
  using CircleBuffers_t = flatbuffers::Vector<flatbuffers::Offset<circle::Buffer>>;
  using CircleTensors_t = flatbuffers::Vector<flatbuffers::Offset<circle::Tensor>>;
  using CircleOperators_t = flatbuffers::Vector<flatbuffers::Offset<circle::Operator>>;

public:
  CircleReader() = default;

public:
  const std::vector<const circle::OperatorCode *> &opcodes() const { return _op_codes; }
  const CircleBuffers_t *buffers() const { return _buffers; }
  const CircleTensors_t *tensors() const { return _tensors; }
  const CircleOperators_t *operators() const { return _operators; }
  const std::vector<int32_t> &inputs() const { return _inputs; }
  const std::vector<int32_t> &outputs() const { return _outputs; }

  uint32_t num_subgraph() const { return _subgraphs->Length(); }

  size_t buffer_info(uint32_t buf_idx, const uint8_t **buff_data);
  circle::BuiltinOperator builtin_code(const circle::Operator *op) const;
  std::string opcode_name(const circle::Operator *op) const;

public:
  bool parse(const circle::Model *model);
  bool select_subgraph(uint32_t subgraph);

private:
  const circle::Model *_model{nullptr};

  const CircleSubGraphs_t *_subgraphs{nullptr};
  const CircleBuffers_t *_buffers{nullptr};
  const CircleTensors_t *_tensors{nullptr};
  const CircleOperators_t *_operators{nullptr};

  std::vector<const circle::OperatorCode *> _op_codes;
  std::vector<int32_t> _inputs;
  std::vector<int32_t> _outputs;
};

} // namespace luci

#endif // __LUCI_IMPORT_GRAPHREADER_H__
