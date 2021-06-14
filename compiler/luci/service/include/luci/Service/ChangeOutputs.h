/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __LUCI_SVC_CHANGE_OUTPUTS_H__
#define __LUCI_SVC_CHANGE_OUTPUTS_H__

#include <loco/IR/Graph.h>

#include <string>
#include <vector>

namespace luci
{

/**
 * @brief Change output to nodes with string name.
 *
 * @note  Should match existing number of nodes and all names should exist.
 *        Will throw exception if failed.
 */
void change_outputs(loco::Graph *, const std::vector<std::string> &);

} // namespace luci

#endif // __LUCI_SVC_CHANGE_OUTPUTS_H__
