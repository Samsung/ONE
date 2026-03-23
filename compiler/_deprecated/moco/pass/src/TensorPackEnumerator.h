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

#ifndef __MOCO_TENSOR_PACK_ENUMERATOR_H__
#define __MOCO_TENSOR_PACK_ENUMERATOR_H__

#include <loco/IR/TensorShape.h>

#include <vector>

namespace moco
{

using u32v_t = std::vector<uint32_t>;

class TensorPackEnumerator
{
public:
  TensorPackEnumerator() = default;

public:
  void shape(const loco::TensorShape &si, const loco::TensorShape &so);
  void axis(uint32_t axis) { _axis = axis; }

public:
  void start(void);
  bool valid(void);
  void advance(void);

public:
  uint32_t inp_num(void) const;
  uint32_t inp_element(void) const;
  uint32_t out_element(void) const;

private:
  void increment(uint32_t);

private:
  loco::TensorShape _shape_inp;
  loco::TensorShape _shape_out;

  uint32_t _axis = 0;
  uint32_t _rank_out = 0;
  uint32_t _num_inp = 0;
  u32v_t _cursor_inp;
  u32v_t _cursor_out;
  u32v_t _boundary_inp;
  u32v_t _boundary_out;
};

} // namespace moco

#endif // __MOCO_TENSOR_PACK_ENUMERATOR_H__
