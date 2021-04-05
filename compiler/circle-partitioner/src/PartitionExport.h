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

#ifndef __CIRCLE_PARTITION_EXPORT_H__
#define __CIRCLE_PARTITION_EXPORT_H__

#include <luci/Partition.h>

#include <string>

namespace partee
{

/**
 * @brief This will save partition connection to json format file
 */
bool export_part_conn_json(const std::string &output_base, const std::string &input,
                           const luci::Module *source, luci::PartedModules &pms);

/**
 * @brief This will save partition connection to ini format file
 */
bool export_part_conn_ini(const std::string &output_base, const std::string &input,
                          const luci::Module *source, luci::PartedModules &pms);

} // namespace partee

#endif // __CIRCLE_PARTITION_EXPORT_H__
