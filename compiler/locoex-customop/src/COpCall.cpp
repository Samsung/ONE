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

#include "locoex/COpCall.h"

#include "locoex/COpAttrTypes.h"

namespace locoex
{

template <COpAttrType AT>
const typename AttrTypeTrait<AT>::Type *COpCall::attr(const std::string &attr_name) const
{
  COpAttrData *attr_data;
  auto found = _attrs.find(attr_name);
  if (found != _attrs.end())
  {
    attr_data = found->second.get();
    return dynamic_cast<const typename AttrTypeTrait<AT>::Type *>(attr_data);
  }
  else
    throw std::runtime_error("Cannot find requested attr");
}

void COpCall::attr(const std::string &attr_name, std::unique_ptr<COpAttrData> &&attr_data)
{
  if (_attrs.find(attr_name) == _attrs.end())
    _attrs[attr_name] = std::move(attr_data);
  else
    throw std::runtime_error("Attr already inserted");
}

std::vector<std::string> COpCall::attr_names() const
{
  std::vector<std::string> attr_names;

  for (auto it = _attrs.cbegin(); it != _attrs.cend(); ++it)
  {
    attr_names.emplace_back(it->first);
  }

  return attr_names;
}

#define INSTANTIATE(AT)                                                                            \
  template const typename AttrTypeTrait<AT>::Type *COpCall::attr<AT>(const std::string &attr_name) \
    const;

INSTANTIATE(COpAttrType::Float)
INSTANTIATE(COpAttrType::Int)

#undef INSTANTIATE

} // namespace locoex
