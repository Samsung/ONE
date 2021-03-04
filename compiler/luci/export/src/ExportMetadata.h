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

#ifndef __LUCI_EXPORT_METADATA_H__
#define __LUCI_EXPORT_METADATA_H__

#include <cassert>
#include <map>
#include <string>
#include <set>
#include <vector>

namespace luci
{

class ExportMetadata
{
public:
  ExportMetadata() = default;

public:
  void link_source(uint32_t node_id, uint32_t source_id);
  void add_source(uint32_t source_id, std::string origin_name);

public:
  const std::map<uint32_t, std::string> &source_table(void) const { return _source_table; }

  const std::map<uint32_t, std::set<uint32_t>> &op_table(void) const { return _op_table; }

private:
  std::map<uint32_t, std::string> _source_table;
  std::map<uint32_t, std::set<uint32_t>> _op_table;
};

const std::vector<uint8_t> encoded_source_table(ExportMetadata &metadata);
const std::vector<uint8_t> encoded_op_table(ExportMetadata &metadata);

} // namespace luci

#endif // __LUCI_EXPORT_METADATA_H__
