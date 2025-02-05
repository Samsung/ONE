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

#ifndef __ONERT_RUN_ALLOCATION_H__
#define __ONERT_RUN_ALLOCATION_H__

#include <cassert>
#include <cstdlib>
#include <cstdint>

#include "nnfw.h"

namespace onert_run
{
class Allocation
{
public:
  Allocation() : data_(nullptr) {}
  ~Allocation() { free(data_); }
  void *data() const { return data_; }
  void *alloc(uint64_t size, NNFW_TYPE dtype)
  {
    size_ = size;
    type_ = dtype;

    assert(data_ == nullptr);
    return data_ = malloc(size);
  }
  uint64_t size() const { return size_; }
  NNFW_TYPE type() const { return type_; }

private:
  void *data_ = nullptr;
  uint64_t size_ = 0;
  NNFW_TYPE type_ = NNFW_TYPE_TENSOR_FLOAT32;
};
} // namespace onert_run

#endif // __ONERT_RUN_ALLOCATION_H__
