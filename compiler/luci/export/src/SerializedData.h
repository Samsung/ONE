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

#ifndef __SERIALIZED_DATA_H__
#define __SERIALIZED_DATA_H__

#include <loco.h>

#include <mio/circle/schema_generated.h>

#include <vector>

#include <unordered_map>

namespace luci
{

struct OpCode
{
  circle::BuiltinOperator opcode;

  bool operator==(const OpCode &rhs) const { return opcode == rhs.opcode; }
};

} // namespace luci

namespace std
{

template <> struct hash<luci::OpCode>
{
  size_t operator()(const luci::OpCode &x) const { return hash<int>()(x.opcode); }
};

} // namespace std

namespace luci
{

/**
 * @breif Record the information of T/F Lite SubGraph and its mapping to loco
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

} // namespace luci

#endif // __SERIALIZED_DATA_H__
