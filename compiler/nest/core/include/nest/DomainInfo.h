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

#ifndef __NEST_DOMAIN_INFO_H__
#define __NEST_DOMAIN_INFO_H__

#include <initializer_list>
#include <vector>

#include <cstdint>

namespace nest
{

class DomainInfo
{
public:
  DomainInfo(std::initializer_list<uint32_t> dims) : _dims{dims}
  {
    // DO NOTHING
  }

public:
  uint32_t rank(void) const { return _dims.size(); }

public:
  uint32_t dim(uint32_t axis) const { return _dims.at(axis); }

private:
  std::vector<uint32_t> _dims;
};

} // namespace nest

#endif // __NEST_DOMAIN_INFO_H__
