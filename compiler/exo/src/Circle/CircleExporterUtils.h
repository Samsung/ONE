/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __CIRCLE_EXPORTER_UTILS_H__
#define __CIRCLE_EXPORTER_UTILS_H__

#include "ExporterUtils.h"

#include "circle_schema_generated.h"

#include "Dialect/IR/TFLNodes.h"

#include <loco.h>

#include <unordered_map>

namespace exo
{
namespace circle_detail
{

struct OpCode
{
  circle::BuiltinOperator opcode;

  bool operator==(const OpCode &rhs) const { return opcode == rhs.opcode; }
};

} // namespace circle_detail
} // namespace exo

namespace exo
{

circle::ActivationFunctionType to_circle_actfunc(locoex::FusedActFunc func);

} // namespace exo

namespace std
{

template <> struct hash<exo::circle_detail::OpCode>
{
  size_t operator()(const exo::circle_detail::OpCode &x) const { return hash<int>()(x.opcode); }
};

} // namespace std

namespace exo
{
namespace circle_detail
{

/**
 * @brief Record the information of T/F Lite SubGraph and its mapping to loco
 */
struct SubGraphContext
{
  /// @brief SubGraph input tensor id
  std::vector<int32_t> _inputs;
  /// @brief SubGraph output tensor id
  std::vector<int32_t> _outputs;
  /// @DataFormat for SubGraph
  circle::DataFormat _data_format{circle::DataFormat::DataFormat_CHANNELS_LAST};
};

// Prerequisites for circle::Model object creation
struct SerializedModelData final : public SubGraphContext
{
  SerializedModelData() = default;
  SerializedModelData(const SerializedModelData &) = delete;

  std::unordered_map<OpCode, uint32_t> _operator_codes;
  std::unordered_map<OpCode, std::string> _custom_operator_codes;
  std::vector<flatbuffers::Offset<circle::Operator>> _operators;
  std::vector<flatbuffers::Offset<circle::Tensor>> _tensors;
  std::vector<flatbuffers::Offset<circle::Buffer>> _buffers;

  // Graph input and output names
  std::unordered_map<loco::Pull *, std::string> _pull_to_name;
  std::unordered_map<loco::Push *, std::string> _push_to_name;

  /**
   * @brief if opcode is not registered in table of opcodes add it
   * @param builtin_code
   * @return idx of opcode in table of opcodes (see schema)
   */
  uint32_t registerBuiltinOpcode(circle::BuiltinOperator builtin_code);
  uint32_t registerCustomOpcode(const std::string &custom_op);
};

circle::Padding getOpPadding(const loco::Padding2D *pad, const loco::Stride<2> *stride,
                             const ShapeDescription &ifm, const ShapeDescription &ofm);
circle::Padding getOpPadding(const locoex::Padding pad);

/// @brief Register graph input and output names to SerializedModelData
void registerGraphIOName(loco::Graph *graph, SerializedModelData &gd);

using TFLTensorIndex = int32_t;

void set_tensor_index(loco::Node *node, const TFLTensorIndex &tensor_id);
TFLTensorIndex get_tensor_index(loco::Node *node);

} // namespace circle_detail
} // namespace exo

#endif // __TFL_EXPORTER_UTILS_H__
