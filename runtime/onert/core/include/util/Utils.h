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

/**
 * @file     Utils.h
 * @brief    This file contains utility macro
 */

#ifndef __ONERT_UTIL_UTILS_H__
#define __ONERT_UTIL_UTILS_H__

#define UNUSED_RELEASE(a) (void)(a)
#include "ir/Coordinates.h"
#include "ir/Shape.h"

template <size_t from, size_t to, typename Enable = void> struct ForEachDimension
{
  template <typename L, typename... Args>
  static void unroll(const onert::ir::Shape &shape, onert::ir::Coordinates &coords,
                     L &&lambda_function, Args &&... args)
  {
    static_assert(from < to, "from must not be less than to");
    assert(static_cast<int>(to) <= shape.rank());
    const auto &d = shape.dim(from);

    for (auto v = 0; v < d; v++)
    {
      coords.set(from, v);
      ForEachDimension<from + 1, to>::unroll(shape, coords, std::forward<L>(lambda_function),
                                             std::forward<Args>(args)...);
    }
  }
};

template <size_t from, size_t to>
struct ForEachDimension<from, to, typename std::enable_if<from == to>::type>
{
  template <typename L, typename... Args>
  static void unroll(const onert::ir::Shape &shape, onert::ir::Coordinates &coords,
                     L &&lambda_function, Args &&... args)
  {
    UNUSED_RELEASE(shape);
    assert(static_cast<int>(to) <= shape.rank());
    lambda_function(coords, std::forward<Args>(args)...);
  }
};

template <typename L, typename... Args>
inline void ShapeLoop(const onert::ir::Shape &shape, L &&lambda_function, Args &&... args)
{
  assert(shape.rank() > 0);
  for (auto i = 0; i < shape.rank(); ++i)
  {
    assert(shape.dim(i) > 0);
  }

  onert::ir::Coordinates coords;
  switch (shape.rank())
  {
    case 0:
      coords.set(0, 0);
      ForEachDimension<0, 0>::unroll(shape, coords, std::forward<L>(lambda_function),
                                     std::forward<Args>(args)...);
      break;
    case 1:
      ForEachDimension<0, 1>::unroll(shape, coords, std::forward<L>(lambda_function),
                                     std::forward<Args>(args)...);
      break;
    case 2:
      ForEachDimension<0, 2>::unroll(shape, coords, std::forward<L>(lambda_function),
                                     std::forward<Args>(args)...);
      break;
    case 3:
      ForEachDimension<0, 3>::unroll(shape, coords, std::forward<L>(lambda_function),
                                     std::forward<Args>(args)...);
      break;
    case 4:
      ForEachDimension<0, 4>::unroll(shape, coords, std::forward<L>(lambda_function),
                                     std::forward<Args>(args)...);
      break;
    case 5:
      ForEachDimension<0, 5>::unroll(shape, coords, std::forward<L>(lambda_function),
                                     std::forward<Args>(args)...);
      break;
    case 6:
      ForEachDimension<0, 6>::unroll(shape, coords, std::forward<L>(lambda_function),
                                     std::forward<Args>(args)...);
      break;
    default:
      assert(false && "ShapeLoop, 1 <= Shape'rank <= 6");
      break;
  }
}
#endif // __ONERT_UTIL_UTILS_H__
