/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci_interpreter/SimpleMemoryManager.h"

namespace luci_interpreter
{

void SimpleMemoryManager::allocate_memory(luci_interpreter::Tensor &tensor)
{
  if (!tensor.is_allocatable())
  {
    return;
  }
  if (tensor.is_data_allocated())
  {
    release_memory(tensor);
  }
  size_t bytes_to_allocate = 0;
  if (tensor.get_raw_size() > 0)
  {
    bytes_to_allocate = tensor.get_raw_size();
  }
  else
  {
    const auto element_size = getDataTypeSize(tensor.element_type());

    // Use large_num_elements to avoid overflow
    const auto num_elements = tensor.shape().large_num_elements();
    bytes_to_allocate = num_elements * element_size;
  }

  auto *data = new uint8_t[bytes_to_allocate];
  tensor.set_data_buffer(data);
}

void SimpleMemoryManager::release_memory(luci_interpreter::Tensor &tensor)
{
  if (!tensor.is_data_allocated())
  {
    tensor.set_data_buffer(nullptr);
    return;
  }
  auto data = tensor.data<uint8_t>();
  delete[] data;
  tensor.set_data_buffer(nullptr);
}

} // namespace luci_interpreter
