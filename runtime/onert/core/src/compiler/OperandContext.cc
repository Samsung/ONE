/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "OperandContext.h"

#include <cassert>

namespace onert
{
namespace compiler
{

OperandContext &OperandContext::set(const ir::OperandIndex &id,
                                    const std::shared_ptr<backend::ITensor> &tensor)
{
  // Only one tensor for an id
  assert(_tensors.find(id) == _tensors.end());
  _tensors[id] = tensor;
  return (*this);
}

void OperandContext::iterate(
    const std::function<void(const ir::OperandIndex &, backend::ITensor &)> &fn)
{
  for (auto &e : _tensors)
  {
    fn(e.first, *e.second);
  }
}

} // namespace compiler
} // namespace onert
