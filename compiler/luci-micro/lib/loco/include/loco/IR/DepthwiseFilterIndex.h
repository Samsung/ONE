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

#ifndef __LOCO_IR_DEPTHWISE_FILTER_INDEX_H__
#define __LOCO_IR_DEPTHWISE_FILTER_INDEX_H__

#include <cstdint>

namespace loco
{

/**
 * @brief DepthwiseFilter Index
 *
 * DepthwiseFilter Index indicates an "element" in a given Depthwise convolution filter.
 *
 * Assume there is a filter K where KS denotes its shape (of DepthwiseFilterShape type).
 *
 * Then, any valid filter index I satisfies the following invariants:
 * - 0 <= I.channel() < KS.depth()
 * - 0 <= I.nth()     < KS.multiplier()
 * - 0 <= I.row()     < KS.height()
 * - 0 <= I.column()  < KS.width()
 */
class DepthwiseFilterIndex final
{
public:
  DepthwiseFilterIndex() = default;

public:
  const uint32_t &channel(void) const { return _channel; }
  uint32_t &channel(void) { return _channel; }

  const uint32_t &nth(void) const { return _nth; }
  uint32_t &nth(void) { return _nth; }

  const uint32_t &row(void) const { return _row; }
  uint32_t &row(void) { return _row; }

  const uint32_t &column(void) const { return _column; }
  uint32_t &column(void) { return _column; }

private:
  uint32_t _channel = 0;
  uint32_t _nth = 0;
  uint32_t _row = 0;
  uint32_t _column = 0;
};

} // namespace loco

#endif // __LOCO_IR_DEPTHWISE_FILTER_INDEX_H__
