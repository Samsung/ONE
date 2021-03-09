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

#include <mio/circle/schema_generated.h>

#include <luci/IR/CircleNodes.h>

#include <vector>

#include <unordered_map>
#include <map>

namespace luci
{

struct OpCode
{
  circle::BuiltinOperator opcode;
  std::string custom_code{""};
  int32_t version = 1;

  bool operator==(const OpCode &rhs) const
  {
    if (opcode == circle::BuiltinOperator_CUSTOM)
    {
      return custom_code == rhs.custom_code;
    }
    else
    {
      return opcode == rhs.opcode;
    }
  }
};

class CircleExportMetadata
{
public:
  void add_source_table(uint32_t source_id, std::string origin_name)
  {
    // Model with multiple subgraph may have different origin_name
    // even if source_id is same. However, as we do not consider about
    // multiple subgraph in profiling for now, just do not care those cases
    // and support them correctly in the future.
    _source_table.emplace(source_id, origin_name);
  }

  void add_op_table(uint32_t node_id, uint32_t source_id)
  {
    // Model with multiple subgraph may have duplicated node id.
    // For now, as we do not consider about multiple subgraph in profiling,
    // just ignore those cases and support them in the future.
    if (_op_table.find(node_id) == _op_table.end())
      _op_table.emplace(node_id, std::set<uint32_t>());
    _op_table.at(node_id).emplace(source_id);
  }

public:
  const std::vector<uint8_t> encoded_source_table(void);
  const std::vector<uint8_t> encoded_op_table(void);

private:
  std::map<uint32_t, std::string> _source_table;
  std::map<uint32_t, std::set<uint32_t>> _op_table;
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
 * @brief Record the information of T/F Lite SubGraph and its mapping to loco
 */
struct SubGraphContext
{
  /// @brief SubGraph name
  std::string _name;
  /// @brief SubGraph input tensor id
  std::vector<int32_t> _inputs;
  /// @brief SubGraph output tensor id
  std::vector<int32_t> _outputs;
  /// @brief DataFormat for SubGraph
  circle::DataFormat _data_format{circle::DataFormat::DataFormat_CHANNELS_LAST};
};

// Prerequisites for circle::Model object creation
struct SerializedModelData final
{
  SerializedModelData() = default;
  SerializedModelData(const SerializedModelData &) = delete;

  std::unordered_map<OpCode, uint32_t> _operator_codes;
  std::vector<flatbuffers::Offset<circle::Buffer>> _buffers;
  CircleExportMetadata _metadata;

  // This is used for removing buffers with same values
  std::map<luci::CircleConst *, uint32_t> _cached_buffer_id;

  /**
   * @brief if opcode is not registered in table of opcodes add it
   * @param builtin_code
   * @return idx of opcode in table of opcodes (see schema)
   */
  uint32_t registerBuiltinOpcode(circle::BuiltinOperator builtin_code, const int32_t op_version);
  uint32_t registerCustomOpcode(const std::string &custom_op);
};

// Prerequisites for circle::Model object creation
struct SerializedGraphData final : public SubGraphContext
{
  SerializedGraphData() = default;
  SerializedGraphData(const SerializedModelData &) = delete;

  std::vector<flatbuffers::Offset<circle::Operator>> _operators;
  std::vector<flatbuffers::Offset<circle::Tensor>> _tensors;
};

} // namespace luci

#endif // __SERIALIZED_DATA_H__
