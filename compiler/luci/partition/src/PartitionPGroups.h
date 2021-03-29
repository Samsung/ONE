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

#ifndef __LUCI_PARTITON_PGROUPS_H__
#define __LUCI_PARTITON_PGROUPS_H__

#include "PartitionIR.h"

#include "luci/Partition.h"

#include <luci/IR/Module.h>

namespace luci
{

/**
 * @brief This will produce a PGroups from Module and PartitionTable.
 * @note  Each PGroup will hold one CircleNode and partition key value as group.
 *        Supports only single Graph in the Module for now.
 */
std::unique_ptr<luci::PGroups> produce_pgroups(const luci::Module *source,
                                               const luci::PartitionTable &partition);

} // namespace luci

#endif // __LUCI_PARTITON_PGROUPS_H__
