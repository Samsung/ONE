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

#ifndef __NEST_VAR_CONTEXT_H__
#define __NEST_VAR_CONTEXT_H__

#include "nest/Bound.h"
#include "nest/Var.h"

#include <vector>

namespace nest
{

class VarContext
{
public:
  uint32_t count(void) const;

public:
  Var make(void);

public:
  Bound &bound(const Var &);
  const Bound &bound(const Var &) const;

private:
  std::vector<Bound> _bound;
};

} // namespace nest

#endif // __NEST_VAR_CONTEXT_H__
