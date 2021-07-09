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

#include "luci/PartitionValidate.h"

#include <luci/Service/Validate.h>

#include <pepper/csv2vec.h>

#include <iostream>

namespace luci
{

bool validate(luci::PartitionTable &partition)
{
  if (partition.groups.size() == 0)
  {
    std::cerr << "There is no 'backends' information";
    return false;
  }
  if (partition.default_group.empty())
  {
    std::cerr << "There is no 'default' backend information";
    return false;
  }
  if (!pepper::is_one_of<std::string>(partition.default_group, partition.groups))
  {
    std::cerr << "'default' backend is not one of 'backends' item";
    return false;
  }
  for (auto &byopcode : partition.byopcodes)
  {
    if (!pepper::is_one_of<std::string>(byopcode.second, partition.groups))
    {
      std::cerr << "OPCODE " << byopcode.first << " is not assigned to one of 'backends' items";
      return false;
    }
  }
  for (auto &byopname : partition.byopnames)
  {
    if (!pepper::is_one_of<std::string>(byopname.second, partition.groups))
    {
      std::cerr << "OPNAME " << byopname.first << " is not assigned to one of 'backends' items";
      return false;
    }
  }
  return true;
}

} // namespace luci
