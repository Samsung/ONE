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

#ifndef __LUCI_CIRCLE_METADATA_IMPORTER_H__
#define __LUCI_CIRCLE_METADATA_IMPORTER_H__

#include "luci/Import/CircleReader.h"

#include <luci/Profile/CircleNodeOrigin.h>

#include <map>
#include <set>

namespace luci
{

class CircleImportMetadata
{
public:
  CircleImportMetadata(const luci::CircleReader &reader);

public:
  const std::map<uint32_t, std::shared_ptr<CircleNodeOrigin>> origin_table(void);

private:
  std::map<uint32_t, std::string> _source_table;
  std::map<uint32_t, std::set<uint32_t>> _op_table;
};

} // namespace luci

#endif // __LUCI_CIRCLE_METADATA_IMPORTER_H__
