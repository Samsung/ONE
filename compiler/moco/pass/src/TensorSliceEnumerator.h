/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __MOCO_TENSOR_SLICE_ENUMERATOR_H__
#define __MOCO_TENSOR_SLICE_ENUMERATOR_H__

#include <loco/IR/TensorShape.h>

#include <vector>

namespace moco
{

using u32v_t = std::vector<uint32_t>;

class TensorSliceEnumerator
{
public:
  TensorSliceEnumerator() = default;

public:
  void shape(loco::TensorShape &s);
  void begin(u32v_t &b) { _begin = b; }
  void end(u32v_t &e) { _end = e; }

public:
  void start(void);
  bool valid(void);
  void advance(void);

  uint32_t cursor(uint32_t rank) const;
  const u32v_t cursor(void) const { return _cursor; }

private:
  void increment(uint32_t);

private:
  loco::TensorShape _shape_in;

  uint32_t _rank_in = 0;
  u32v_t _cursor;
  u32v_t _boundary;
  u32v_t _begin;
  u32v_t _end;
};

} // namespace moco

#endif // __MOCO_TENSOR_SLICE_ENUMERATOR_H__
