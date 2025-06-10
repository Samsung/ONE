/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "Node.h"

namespace onert::dumper::dot
{

Node::Node(const std::string &id) : _id{id}
{
  // Set default values
  _attributes["style"] = "filled";
  _attributes["colorscheme"] = DEFAULT_COLORSCHEME;
  _attributes["fillcolor"] = DEFAULT_FILLCOLOR;
}

void Node::setAttribute(const std::string &key, const std::string &val) { _attributes[key] = val; }

std::string Node::getAttribute(const std::string &key)
{
  auto itr = _attributes.find(key);
  if (itr == _attributes.end())
  {
    return "";
  }
  else
  {
    return itr->second;
  }
}

} // namespace onert::dumper::dot
