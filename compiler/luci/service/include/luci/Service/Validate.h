/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LUCI_SERVICE_VALIDATE_H__
#define __LUCI_SERVICE_VALIDATE_H__

#include <luci/IR/Module.h>

#include <loco.h>

namespace luci
{

bool validate(loco::Graph *);

/**
 * @brief Return true if all nodes in graph have non empty name
 */
bool validate_name(loco::Graph *);

/**
 * @brief Return true if all names in the Module are unique
 * @note  CircleOutput may have duplicate name
 */
bool validate_unique_name(luci::Module *);

bool validate(luci::Module *);

bool validate_shape(loco::Graph *);

bool validate_shape(luci::Module *);

} // namespace luci

#endif // __LUCI_SERVICE_VALIDATE_H__
