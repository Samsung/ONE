/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_TRAIN_ALLOCATION_H__
#define __ONERT_TRAIN_ALLOCATION_H__

#include <cstdlib>
#include <cstdint>
#include <functional>
#include <vector>

namespace onert_train
{
class Allocation
{
public:
  Allocation() : data_(nullptr) {}
  ~Allocation() { free(data_); }
  void *data() const { return data_; }
  void *alloc(uint64_t sz)
  {
    if (data_)
    {
      free(data_);
    }

    return data_ = malloc(sz);
  }

private:
  void *data_;
};

using Generator = std::function<bool(uint32_t,                  /** index **/
                                     std::vector<Allocation> &, /** input **/
                                     std::vector<Allocation> & /** expected **/)>;

} // namespace onert_train

#endif // __ONERT_TRAIN_ALLOCATION_H__
