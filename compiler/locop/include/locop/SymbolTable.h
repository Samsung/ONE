/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LOCOP_SYMBOL_TABLE_H__
#define __LOCOP_SYMBOL_TABLE_H__

#include <loco.h>

#include <string>

namespace locop
{

/**
 * @brief Symbol Table Interface
 *
 * Symbol Table gives a name for each node.
 */
struct SymbolTable
{
  virtual ~SymbolTable() = default;

  virtual std::string lookup(const loco::Node *) const = 0;
};

} // namespace locop

#endif // __LOCOP_SYMBOL_TABLE_H__
