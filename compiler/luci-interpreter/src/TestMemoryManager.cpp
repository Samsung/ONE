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

#include "luci_interpreter/TestMemoryManager.h"

namespace luci_interpreter
{

void TestMemoryManager::allocate_memory(luci_interpreter::Tensor &tensor)
{
  if (!tensor.is_allocatable())
  {
    return;
  }
  if (tensor.is_data_allocated())
  {
    release_memory(tensor);
  }
  const auto element_size = getDataTypeSize(tensor.element_type());
  const auto num_elements = tensor.shape().num_elements();

  auto *data = new uint8_t[num_elements * element_size];
  allocations.push_back(data);
  tensor.set_data_buffer(data);
}

void TestMemoryManager::release_memory(luci_interpreter::Tensor &tensor)
{
  tensor.set_data_buffer(nullptr);
}

} // namespace luci_interpreter
