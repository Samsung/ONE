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

#ifndef __NEST_SCHEDULE_H__
#define __NEST_SCHEDULE_H__

#include "nest/Module.h"
#include "nest/Level.h"

#include <vector>

namespace nest
{

class Schedule final
{
public:
  explicit Schedule(const Module &);

public:
  const VarContext &var(void) const { return _module.var(); }
  const DomainContext &domain(void) const { return _module.domain(); }
  const Block &block(void) const { return _module.block(); }
  const Ret &ret(void) const { return _module.ret(); }

public:
  Var at(const Level &) const;
  Level level(const Var &) const;

private:
  Module _module;

private:
  std::vector<VarID> _level;
};

} // namespace nest

#endif // __NEST_SCHEDULE_H___
