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

#include "moco/Import/ModelSignature.h"

#include <oops/UserExn.h>

namespace moco
{

void ModelSignature::add_customop(const std::string &op)
{
  if (std::find(_customops.begin(), _customops.end(), op) == _customops.end())
    _customops.emplace_back(op);
  else
    throw oops::UserExn("Duplicate custom operator", op);
}

void ModelSignature::shape(const std::string &node_name, const angkor::TensorShape &shape)
{
  if (_shapes.find(node_name) != _shapes.end())
    throw oops::UserExn("Duplicate node name", node_name);

  _shapes[node_name] = shape;
}

const angkor::TensorShape *ModelSignature::shape(const std::string &node_name) const
{
  auto res = _shapes.find(node_name);
  if (res == _shapes.end())
    return nullptr;
  else
    return &res->second;
}

void ModelSignature::dtype(const std::string &node_name, loco::DataType dtype)
{
  if (_dtypes.find(node_name) != _dtypes.end())
    throw oops::UserExn("Duplicate node name", node_name);

  _dtypes[node_name] = dtype;
}

loco::DataType ModelSignature::dtype(const std::string &node_name) const
{
  auto res = _dtypes.find(node_name);
  if (res == _dtypes.end())
    return loco::DataType::Unknown;
  else
    return res->second;
}

} // namespace moco
