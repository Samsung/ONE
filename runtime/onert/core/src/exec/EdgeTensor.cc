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

#include "EdgeTensor.h"

namespace onert::exec
{

bool EdgeTensor::applyShape(const ir::Shape &new_shape)
{
  bool previously_dynamic = is_dynamic();
  if (!previously_dynamic || _buffer == nullptr)
  {
    // Always set shape - when buffer with same size was already allocated, shape could differ
    setShape(new_shape);
    set_dynamic();
    const auto total_size = get_info().total_size();
    _buffer = std::make_unique<uint8_t[]>(total_size);
  }
  else
  {
    auto previous_size = total_size();
    auto new_size = new_shape.num_elements() * ir::sizeOfDataType(data_type());
    if (previous_size != new_size)
    {
      setShape(new_shape);
      set_dynamic();
      const auto total_size = get_info().total_size();
      _buffer = std::make_unique<uint8_t[]>(total_size);
    }
    else
    { // when buffer with same size was already allocated, shape could differ
      setShape(new_shape);
    }
  }
  return true;
}

} // namespace onert::exec
