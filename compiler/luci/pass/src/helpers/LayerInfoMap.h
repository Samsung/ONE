/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LUCI_PASS_HELPERS_LAYER_INFO_MAP_H__
#define __LUCI_PASS_HELPERS_LAYER_INFO_MAP_H__

#include <luci/Pass/QuantizationParameters.h>

#include <unordered_map>

namespace luci
{

using LayerInfoMap = std::unordered_map<std::string, luci::LayerInfo *>;

LayerInfoMap layer_info_map(loco::Graph *g, std::vector<LayerInfo> &layers_info);

} // namespace luci

#endif // __LUCI_PASS_HELPERS_LAYER_INFO_MAP_H__
