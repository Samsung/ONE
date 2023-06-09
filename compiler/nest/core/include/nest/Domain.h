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

#ifndef __NEST_DOMAIN_H__
#define __NEST_DOMAIN_H__

#include "nest/Closure.h"

namespace nest
{

class Domain
{
public:
  Domain() = default;

public:
  Domain(const DomainID &id) : _id{id}
  {
    // DO NOTHING
  }

public:
  Domain(const Domain &) = default;

public:
  template <typename... Args> Closure operator()(Args &&...indices)
  {
    return Closure{_id, std::forward<Args>(indices)...};
  }

public:
  const DomainID &id(void) const { return _id; }

private:
  DomainID const _id;
};

} // namespace nest

#endif // __NEST_DOMAIN_H__
