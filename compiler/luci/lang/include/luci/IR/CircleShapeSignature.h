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

#ifndef __LUCI_IR_SHAPE_SIGNATURE_H__
#define __LUCI_IR_SHAPE_SIGNATURE_H__

#include <stdint.h>
#include <vector>

namespace luci
{

class ShapeSignature
{
public:
  ShapeSignature() = default;

  ShapeSignature(const std::vector<int32_t> &shape_signature)
  {
    _shape_signature = shape_signature;
  }

  const std::vector<int32_t> as_vector() const { return _shape_signature; }

  const int32_t &dim(uint32_t d) const { return _shape_signature.at(d); }
  int32_t &dim(uint32_t d) { return _shape_signature.at(d); }

  uint32_t rank(void) const { return _shape_signature.size(); }
  void rank(uint32_t rank) { _shape_signature.resize(rank); }

private:
  std::vector<int32_t> _shape_signature{};
};

} // namespace luci

#endif // __LUCI_IR_SHAPE_SIGNATURE_H__
