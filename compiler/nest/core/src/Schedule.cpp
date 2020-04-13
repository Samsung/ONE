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

#include "nest/Schedule.h"

#include <cassert>
#include <stdexcept>

namespace nest
{

Schedule::Schedule(const Module &module) : _module{module}
{
  // NOTE This implementation assumes that VarContext sequentially assigns VarID
  for (uint32_t n = 0; n < _module.var().count(); ++n)
  {
    _level.emplace_back(VarID{n});
  }

  assert(_level.size() == _module.var().count());
}

Var Schedule::at(const Level &lv) const { return Var{_level.at(lv.value())}; }

Level Schedule::level(const Var &var) const
{
  for (uint32_t lv = 0; lv < _level.size(); ++lv)
  {
    if (_level.at(lv) == var.id())
    {
      return Level{lv};
    }
  }

  throw std::invalid_argument{"var"};
}

} // namespace nest
