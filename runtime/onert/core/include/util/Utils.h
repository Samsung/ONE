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

#include "ir/Coordinates.h"
#include "ir/Shape.h"

#define UNUSED_RELEASE(a) (void)(a)

template <size_t rest> struct ForEachDimension
{
  template <typename L>
  static void unroll(const onert::ir::Shape &shape, onert::ir::Coordinates &coords,
                     L lambda_function)
  {
    if (static_cast<int>(rest) > shape.rank())
    {
      ForEachDimension<rest - 1>::unroll(shape, coords, lambda_function);
      return;
    }

    const auto axis = shape.rank() - rest;
    const auto &d = shape.dim(axis);

    for (auto v = 0; v < d; v++)
    {
      coords.set(axis, v);
      ForEachDimension<rest - 1>::unroll(shape, coords, lambda_function);
    }
  }
};

template <> struct ForEachDimension<0>
{
  template <typename L>
  static void unroll(const onert::ir::Shape &shape, onert::ir::Coordinates &coords,
                     L lambda_function)
  {
    UNUSED_RELEASE(shape);
    lambda_function(coords);
  }
};

template <typename L> inline void ShapeLoop(const onert::ir::Shape &shape, L lambda_function)
{
  int32_t rank = shape.rank();
  assert(rank > 0);
  for (int32_t i = 0; i < rank; ++i)
  {
    assert(shape.dim(i) > 0);
  }

  onert::ir::Coordinates coords;
  if (rank == 0)
  {
    coords.set(0, 0);
  }
  // TODO Change 6 to onert::ir::Shape::kMaxRank if onert::ir::Shape::kMaxRank is modified as a
  // constant expression
  ForEachDimension<6>::unroll(shape, coords, lambda_function);
}
#endif // __ONERT_UTIL_UTILS_H__
