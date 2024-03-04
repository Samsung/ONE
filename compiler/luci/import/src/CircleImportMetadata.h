/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LUCI_CIRCLE_IMPORT_METADATA_H__
#define __LUCI_CIRCLE_IMPORT_METADATA_H__

#include "luci/Import/CircleReader.h"

#include <luci/Profile/CircleNodeOrigin.h>
#include <luci/IR/ExecutionPlanTable.h>

#include <map>
#include <set>
#include <string>

namespace luci
{

using OriginTable = std::map<uint32_t, std::shared_ptr<CircleNodeOrigin>>;

class CircleImportMetadata
{
public:
  CircleImportMetadata() = delete;

  CircleImportMetadata(const luci::CircleReader &reader);

public:
  /**
   * @brief Create origin table using _source_table and _op_table in CircleImportMetadata
   * @note  For creating origin table, both _op_table and _source_table should exist.
   *        If one of them does not exist, empty table is returned.
   */
  const OriginTable origin_table(void);

  const std::map<uint32_t, std::string> &source_table(void) const { return _source_table; }

  const std::map<uint32_t, std::uint32_t> &map_tensors_indexes(void) const { return _map_tensors_indexes; }

  const luci::ExecutionPlanTable &execution_plan_table(void) const { return _execution_plan_table; }

private:
  // Decoded metadata is stored
  std::map<uint32_t, std::string> _source_table;
  std::map<uint32_t, std::set<uint32_t>> _op_table;
  // _execution_plan_table stores for node with node_id order of execution,
  // and offsets output tensors
  luci::ExecutionPlanTable _execution_plan_table;
  std::map<uint32_t, uint32_t> _map_tensors_indexes;
};

} // namespace luci

#endif // __LUCI_CIRCLE_IMPORT_METADATA_H__
