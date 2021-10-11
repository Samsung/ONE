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

bool is_valid(const circle::OperatorCodeT &opcode);
bool is_valid(const circle::OperatorCode *opcode);
bool is_custom(const circle::OperatorCodeT &opcode);
bool is_custom(const circle::OperatorCode *opcode);
std::string opcode_name(const circle::OperatorCodeT &opcode);
std::string opcode_name(const circle::OperatorCode *opcode);
const char *tensor_name(const circle::TensorT &tensor);
const char *tensor_name(const circle::Tensor *tensor);
const circle::QuantizationParametersT *tensor_quantization(const circle::TensorT &tensor);
const circle::QuantizationParameters *tensor_quantization(const circle::Tensor *tensor);

loco::DataType luci_datatype(circle::TensorType type);
FusedActFunc luci_actfunc(const circle::ActivationFunctionType type);
Padding luci_padding(const circle::Padding padding);
MirrorPadMode luci_mirrorpad_mode(const circle::MirrorPadMode mode);
luci::CircleFullyConnected::WeightsFormat
luci_weights_format(const circle::FullyConnectedOptionsWeightsFormat weights_format);
std::unique_ptr<CircleQuantParam>
luci_quantparam(const circle::QuantizationParametersT *quantization);

/// @brief Copy common tensor attributes such as name, type, etc. to node.
void copy_tensor_attributes(const circle::TensorT &tensor, CircleNode *node);
void copy_tensor_attributes(const circle::Tensor *tensor, CircleNode *node);

template <typename T> class VectorWrapper
{
public:
  explicit VectorWrapper(const flatbuffers::Vector<T> *ptr);

  uint32_t size() const;
  const T *data() const;

  using iterator = typename flatbuffers::Vector<T>::const_iterator;
  iterator begin() const;
  iterator end() const;

  using value_type = typename flatbuffers::Vector<T>::return_type;
  value_type at(uint32_t i) const;
  value_type operator[](uint32_t i) const;

  bool is_null() const;
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
private:
  using CircleBuffers_t = std::vector<std::unique_ptr<circle::BufferT>>;
  using CircleTensors_t = std::vector<std::unique_ptr<circle::TensorT>>;
  using CircleOperators_t = std::vector<std::unique_ptr<circle::OperatorT>>;
  using CircleOperatorCodes_t = std::vector<std::unique_ptr<circle::OperatorCodeT>>;
  using CircleMetadata_t = std::vector<std::unique_ptr<circle::MetadataT>>;

  using CircleSubGraphs = VectorWrapper<flatbuffers::Offset<circle::SubGraph>>;
  using CircleBuffers = VectorWrapper<flatbuffers::Offset<circle::Buffer>>;
  using CircleTensors = VectorWrapper<flatbuffers::Offset<circle::Tensor>>;
  using CircleOperators = VectorWrapper<flatbuffers::Offset<circle::Operator>>;
  using CircleOperatorCodes = VectorWrapper<flatbuffers::Offset<circle::OperatorCode>>;
  using CircleMetadataSet = VectorWrapper<flatbuffers::Offset<circle::Metadata>>;

public:
  CircleReader() = default;

public: // unpacked API
  const CircleOperatorCodes_t &opcodes() const { return _model->operator_codes; }
  const CircleBuffers_t &buffers() const { return _model->buffers; }
  const CircleTensors_t &tensors() const { return _current_subgraph->tensors; }
  const CircleOperators_t &operators() const { return _current_subgraph->operators; }
  const std::vector<int32_t> &inputs() const { return _current_subgraph->inputs; }
  const std::vector<int32_t> &outputs() const { return _current_subgraph->outputs; }
  const std::string &name() const { return _current_subgraph->name; }
  const circle::DataFormat &data_format() const { return _current_subgraph->data_format; }
  const CircleMetadata_t &metadata() const { return _model->metadata; }

  circle::BuiltinOperator builtin_code(const circle::OperatorT &op) const;
  std::string opcode_name(const circle::OperatorT &op) const;

public: // direct API
  CircleOperatorCodes native_opcodes() const { return wrap(_native_model->operator_codes()); }
  CircleBuffers native_buffers() const { return wrap(_native_model->buffers()); }
  CircleTensors native_tensors() const { return wrap(_native_subgraph->tensors()); }
  CircleOperators native_operators() const { return wrap(_native_subgraph->operators()); }
  VectorWrapper<int32_t> native_inputs() const { return wrap(_native_subgraph->inputs()); }
  VectorWrapper<int32_t> native_outputs() const { return wrap(_native_subgraph->outputs()); }
  const char *native_name() const { return _native_subgraph->name()->c_str(); }
  circle::DataFormat native_data_format() const { return _native_subgraph->data_format(); }
  CircleMetadataSet native_metadata() const { return wrap(_native_model->metadata()); }

  circle::BuiltinOperator builtin_code(const circle::Operator *op) const;
  std::string opcode_name(const circle::Operator *op) const;

public:
  uint32_t num_subgraph() const { return _model->subgraphs.size(); }

public:
  bool parse(const circle::Model *model);
  bool select_subgraph(uint32_t subgraph);

private:
  // unpacked model
  std::unique_ptr<const circle::ModelT> _model;
  const circle::SubGraphT *_current_subgraph{nullptr};

  // direct model
  const circle::Model *_native_model{nullptr};
  const circle::SubGraph *_native_subgraph{nullptr};
};

} // namespace luci

#endif // __LUCI_IMPORT_GRAPHREADER_H__
