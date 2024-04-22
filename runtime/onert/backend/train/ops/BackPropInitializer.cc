/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "BackPropInitializer.h"

#include "OperationUtils.h"

namespace onert
{
namespace backend
{
namespace train
{
namespace ops
{

BackPropInitializer::BackPropInitializer(const std::vector<BackPropTensor *> back_props)
  : _back_props{back_props}
{
  assert(std::all_of(back_props.cbegin(), back_props.cend(),
                     [](const BackPropTensor *back_prop) { return back_prop != nullptr; }));
}

void BackPropInitializer::forward(bool)
{
  // DO NOTHING
}

void BackPropInitializer::backward()
{
  for (auto &&back_prop_tensor : _back_props)
  {
    assert(back_prop_tensor->buffer() != nullptr);
    memset(back_prop_tensor->buffer(), 0, back_prop_tensor->total_size());
  }
}

} // namespace ops
} // namespace train
} // namespace backend
} // namespace onert
