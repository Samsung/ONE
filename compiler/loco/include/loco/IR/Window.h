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

#ifndef __LOCO_IR_WINDOW_H__
#define __LOCO_IR_WINDOW_H__

#include <cstdint>

namespace loco
{

/**
 * @brief ND Receptive Field Shape
 *
 * Window<N> describes the shape of N-dimensional receptive field.
 */
template <unsigned N> class Window;

/**
 * @brief 2D Receptive Field Shape
 */
template <> class Window<2> final
{
public:
  uint32_t vertical(void) const { return _vertical; }
  void vertical(uint32_t value) { _vertical = value; }

public:
  uint32_t horizontal(void) const { return _horizontal; }
  void horizontal(uint32_t value) { _horizontal = value; }

private:
  uint32_t _vertical = 1;
  uint32_t _horizontal = 1;
};

} // namespace loco

#endif // __LOCO_IR_WINDOW_H__
