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

#ifndef __LUCI_IMPORT_METADATA_HANDLER_H__
#define __LUCI_IMPORT_METADATA_HANDLER_H__

#include "luci/Import/CircleReader.h"

#include <luci/Profile/CircleNodeOrigin.h>

#include <map>
#include <string>
#include <set>

namespace luci
{

class ImportMetadata
{
public:
  ImportMetadata() = delete;

  ImportMetadata(const luci::CircleReader &reader);

public:
  std::shared_ptr<CircleNodeOrigin> find_origin(uint32_t id) const
  {
    if (_origin_table.find(id) == _origin_table.end())
      throw std::runtime_error("Origin is not found");
    return _origin_table.at(id);
  }

  bool has_origin_data(void) const { return _origin_table.size() > 0; }

private:
  std::map<uint32_t, std::shared_ptr<CircleNodeOrigin>> _origin_table;
};

} // namespace luci

#endif // __LUCI_IMPORT_METADATA_HANDLER_H__
