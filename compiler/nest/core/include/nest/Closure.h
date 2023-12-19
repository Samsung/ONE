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

#ifndef __NEST_CLOSURE_H__
#define __NEST_CLOSURE_H__

#include "nest/DomainID.h"
#include "nest/Expr.h"

namespace nest
{

class Closure
{
public:
  template <typename... Args>
  Closure(const DomainID &id, Args &&...indices) : _id{id}, _sub{std::forward<Args>(indices)...}
  {
    // DO NOTHING
  }

public:
  operator Expr() const;

public:
  const DomainID &id(void) const { return _id; }
  const expr::Subscript &sub(void) const { return _sub; }

private:
  DomainID const _id;
  expr::Subscript const _sub;
};

} // namespace nest

#endif // __NEST_CLOSURE_H__
