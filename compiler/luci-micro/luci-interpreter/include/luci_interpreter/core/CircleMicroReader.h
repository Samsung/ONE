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

#ifndef __LUCI_MICRO_INTERPRETER_MICRO_READER_H__
#define __LUCI_MICRO_INTERPRETER_MICRO_READER_H__

#include "ParamsType.h"
#include "DataType.h"

#include <circle-generated/circle/schema_generated.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace luci_interpreter
{

const char *tensor_name(const circle::Tensor *tensor);

DataType luci_datatype(circle::TensorType type);
FusedActFunc luci_actfunc(circle::ActivationFunctionType type);
Padding luci_padding(circle::Padding padding);
MirrorPadMode luci_mirrorpad_mode(circle::MirrorPadMode mode);

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
  circle::DataFormat data_format() const { return _current_subgraph->data_format(); }
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

} // namespace luci_interpreter

#endif // __LUCI_MICRO_INTERPRETER_MICRO_READER_H__
