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
#include <luci/IR/CircleNode.h>
#include <luci/IR/CircleQuantParam.h>
#include <luci/IR/SparsityParam.h>

#include <loco.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace luci
{

const char *tensor_name(const circle::Tensor *tensor);
const circle::QuantizationParameters *tensor_quantization(const circle::Tensor *tensor);

loco::DataType luci_datatype(circle::TensorType type);
FusedActFunc luci_actfunc(const circle::ActivationFunctionType type);
Padding luci_padding(const circle::Padding padding);
MirrorPadMode luci_mirrorpad_mode(const circle::MirrorPadMode mode);
luci::CircleFullyConnected::WeightsFormat
luci_weights_format(const circle::FullyConnectedOptionsWeightsFormat weights_format);
std::unique_ptr<CircleQuantParam>
luci_quantparam(const circle::QuantizationParameters *quantization);

/// @brief Copy common tensor attributes such as name, type, etc. to node.
void copy_tensor_attributes(const circle::Tensor *tensor, CircleNode *node);

std::string fb_string2std_string(const flatbuffers::String *fb_str);

/**
 * @brief Wrapper to use flatbuffers::Vector pointer as std::vector entity
 */
template <typename T> class VectorWrapper
{
public:
  explicit VectorWrapper(const flatbuffers::Vector<T> *ptr);

  const T *data() const;
  uint32_t size() const;

  using iterator = typename flatbuffers::Vector<T>::const_iterator;
  iterator begin() const;
  iterator end() const;

  using value_type = typename flatbuffers::Vector<T>::return_type;
  value_type at(uint32_t i) const;
  value_type operator[](uint32_t i) const;

  bool null() const;
  bool empty() const;

private:
  const flatbuffers::Vector<T> *_vector;
};

template <typename T> VectorWrapper<T> wrap(const flatbuffers::Vector<T> *vec)
{
  return VectorWrapper<T>(vec);
}

/**
 * @brief Loads Circle file and provides helpers to access attributes
 */
class CircleReader
{
private: // direct API
  using CircleBuffers = VectorWrapper<flatbuffers::Offset<circle::Buffer>>;
  using CircleTensors = VectorWrapper<flatbuffers::Offset<circle::Tensor>>;
  using CircleOperators = VectorWrapper<flatbuffers::Offset<circle::Operator>>;
  using CircleOperatorCodes = VectorWrapper<flatbuffers::Offset<circle::OperatorCode>>;
  using CircleMetadataSet = VectorWrapper<flatbuffers::Offset<circle::Metadata>>;

public:
  CircleReader() = default;

public: // direct API
  CircleOperatorCodes opcodes() const { return wrap(_model->operator_codes()); }
  CircleBuffers buffers() const { return wrap(_model->buffers()); }
  CircleTensors tensors() const { return wrap(_current_subgraph->tensors()); }
  CircleOperators operators() const { return wrap(_current_subgraph->operators()); }
  VectorWrapper<int32_t> inputs() const { return wrap(_current_subgraph->inputs()); }
  VectorWrapper<int32_t> outputs() const { return wrap(_current_subgraph->outputs()); }
  std::string name() const { return fb_string2std_string(_current_subgraph->name()); }
  CircleMetadataSet metadata() const { return wrap(_model->metadata()); }

  uint32_t num_subgraph() const { return wrap(_model->subgraphs()).size(); }

  circle::BuiltinOperator builtin_code(const circle::Operator *op) const;
  std::string opcode_name(const circle::Operator *op) const;

public:
  bool parse(const circle::Model *model);
  bool select_subgraph(uint32_t subgraph);

private:
  const circle::Model *_model{nullptr};
  const circle::SubGraph *_current_subgraph{nullptr};
};

} // namespace luci

#endif // __LUCI_IMPORT_GRAPHREADER_H__
