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

bool is_valid(const circle::OperatorCodeT &opcode);
bool is_custom(const circle::OperatorCodeT &opcode);
std::string opcode_name(const circle::OperatorCodeT &opcode);
const char *tensor_name(const circle::TensorT &tensor);
const circle::QuantizationParametersT *tensor_quantization(const circle::TensorT &tensor);

loco::DataType luci_datatype(circle::TensorType type);
FusedActFunc luci_actfunc(const circle::ActivationFunctionType type);
Padding luci_padding(const circle::Padding padding);
std::unique_ptr<CircleQuantParam>
luci_quantparam(const circle::QuantizationParametersT *quantization);

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

public:
  CircleReader() = default;

public:
  const CircleOperatorCodes_t &opcodes() const { return _model->operator_codes; }
  const CircleBuffers_t &buffers() const { return _model->buffers; }
  const CircleTensors_t &tensors() const { return _current_subgraph->tensors; }
  const CircleOperators_t &operators() const { return _current_subgraph->operators; }
  const std::vector<int32_t> &inputs() const { return _current_subgraph->inputs; }
  const std::vector<int32_t> &outputs() const { return _current_subgraph->outputs; }
  const std::string &name() const { return _current_subgraph->name; }

  uint32_t num_subgraph() const { return _model->subgraphs.size(); }

  circle::BuiltinOperator builtin_code(const circle::OperatorT &op) const;
  std::string opcode_name(const circle::OperatorT &op) const;

public:
  bool parse(const circle::Model *model);
  bool select_subgraph(uint32_t subgraph);

private:
  std::unique_ptr<const circle::ModelT> _model;
  const circle::SubGraphT *_current_subgraph{nullptr};
};

} // namespace luci

#endif // __LUCI_IMPORT_GRAPHREADER_H__
