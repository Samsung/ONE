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

#ifndef __LOCO_IR_PADDINGND_H__
#define __LOCO_IR_PADDINGND_H__

#include <cstdint>
#include <vector>

namespace loco
{

/**
 * This class indicates how many pads to add before(front) and after(back) the contents of
 * tensor in that dimension.
 */
class PaddingND final
{

public:
  const uint32_t &front(uint32_t dim) const { return _front.at(dim); }
  uint32_t &front(uint32_t dim) { return _front.at(dim); }

public:
  const uint32_t &back(uint32_t dim) const { return _back.at(dim); }
  uint32_t &back(uint32_t dim) { return _back.at(dim); }

public:
  uint32_t rank(void) const { return _front.size(); }
  void rank(uint32_t s)
  {
    _front.resize(s);
    _back.resize(s);
  }

private:
  std::vector<uint32_t> _front;
  std::vector<uint32_t> _back;
};

} // namespace loco

#endif // __LOCO_IR_PADDINGND_H__
