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

#ifndef __NEST_BLOCK_H__
#define __NEST_BLOCK_H__

#include "nest/Stmt.h"

#include <vector>

#include <cstdint>

namespace nest
{

struct Block
{
public:
  uint32_t size(void) const { return _stmts.size(); }

public:
  const Stmt &at(uint32_t n) const { return _stmts.at(n); }

public:
  void append(const Stmt &stmt) { _stmts.emplace_back(stmt); }

private:
  std::vector<Stmt> _stmts;
};

} // namespace nest

#endif // __NEST_BLOCK_H__
