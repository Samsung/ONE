/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LUCI_POST_IMPORT_H__
#define __LUCI_POST_IMPORT_H__

#include "luci/Import/CircleReader.h"

#include "luci/IR/Module.h"

namespace luci
{

/**
 * @brief  Do post import actions
 */
void post_import_graph(luci::Module *module, const luci::CircleReader &reader);

} // namespace luci

#endif // __LUCI_POST_IMPORT_H__
