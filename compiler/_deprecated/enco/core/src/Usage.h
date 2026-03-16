/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ENCO_USAGE_H__
#define __ENCO_USAGE_H__

#include "coco/IR.h"

#include <set>

namespace enco
{

/// @brief Returns the set of blocks that reads a given bag
std::set<coco::Block *> readers(const coco::Bag *bag);
/// @brief Return the set of blocks that updates a given bag
std::set<coco::Block *> updaters(const coco::Bag *bag);

} // namespace enco

#endif // __ENCO_USAGE_H__
