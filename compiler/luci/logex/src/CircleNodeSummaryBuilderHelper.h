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

#ifndef __LUCI_LOGEX_CIRCLE_NODE_SUMMARY_BUILDER_HELPER__
#define __LUCI_LOGEX_CIRCLE_NODE_SUMMARY_BUILDER_HELPER__

#include <luci/IR/AttrFusedActFunc.h>
#include <luci/IR/AttrPadding.h>
#include <luci/IR/AttrMirrorPadMode.h>
#include <luci/IR/AttrStride.h>
#include <luci/IR/AttrFilter.h>
#include <luci/IR/CircleOpcode.h>
#include <loco/IR/DataType.h>

#include <string>

namespace luci
{

std::string to_str(bool value);
std::string to_str(loco::DataType type);
std::string to_str(luci::FusedActFunc fused);
std::string to_str(luci::Padding padding);
std::string to_str(luci::MirrorPadMode mode);
std::string to_str(const luci::Stride *stride);
std::string to_str(const luci::Filter *filter);

std::string circle_opname(luci::CircleOpcode opcode);

} // namespace luci

#endif // __LUCI_LOGEX_CIRCLE_NODE_SUMMARY_BUILDER_HELPER__
