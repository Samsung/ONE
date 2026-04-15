/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __COCO_ADT_PTR_LIST_H__
#define __COCO_ADT_PTR_LIST_H__

#include <vector>

#include <cstdint>

namespace coco
{

template <typename T> class PtrList
{
public:
  PtrList() = default;

public:
  PtrList(const PtrList &) = delete;
  PtrList(PtrList &&) = delete;

public:
  virtual ~PtrList() = default;

public:
  uint32_t size(void) const { return _ptrs.size(); }

public:
  T *at(uint32_t n) const { return _ptrs.at(n); }

public:
  void insert(T *ptr) { _ptrs.emplace_back(ptr); }

private:
  std::vector<T *> _ptrs;
};

} // namespace coco

#endif // __COCO_ADT_PTR_LIST_H__
