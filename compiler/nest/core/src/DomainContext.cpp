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

#include "nest/DomainContext.h"

namespace nest
{

uint32_t DomainContext::count(void) const { return _info.size(); }

Domain DomainContext::make(std::initializer_list<uint32_t> dims)
{
  const DomainID domain_id{count()};

  _info.emplace_back(dims);

  return Domain{domain_id};
}

const DomainInfo &DomainContext::info(const Domain &dom) const
{
  return _info.at(dom.id().value());
}

} // namespace nest
