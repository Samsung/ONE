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

#include "nest/Module.h"

#include <cassert>

namespace nest
{

void Module::push(const Expr &expr)
{
  auto stmt = std::make_shared<stmt::PushNode>(expr);
  _block.append(stmt);
}

void Module::ret(const Closure &clo)
{
  // Only one RET is allowed for each module
  assert(_ret == nullptr);
  _ret = std::make_shared<Ret>(clo.id(), clo.sub());
}

const Ret &Module::ret(void) const
{
  // Caller should NOT invoke this method before setting ret
  assert(_ret != nullptr);
  return *_ret;
}

} // namespace nest
