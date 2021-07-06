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

#include "luci/PartitionDump.h"

namespace
{

void dump(std::ostream &os, const luci::PartitionTable &table)
{
  os << "Backends:";
  for (auto &group : table.groups)
  {
    os << " " << group;
    if (table.default_group == group)
      os << "(default)";
  }
  os << std::endl;

  os << "Assign by OPCODE: " << std::endl;
  for (auto &item : table.byopcodes)
    os << "  " << item.first << "=" << item.second << std::endl;

  os << "Assign by OPNAME: " << std::endl;
  for (auto &item : table.byopnames)
    os << "  " << item.first << "=" << item.second << std::endl;
}

} // namespace

std::ostream &operator<<(std::ostream &os, const luci::PartitionTable &table)
{
  dump(os, table);
  return os;
}
