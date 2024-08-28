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

#include "ir/Coordinates.h"

#include <cassert>

namespace onert
{
namespace ir
{

Coordinates convertCoordinates(const Coordinates &from_coordinates, const PermuteType &type)
{
  assert(from_coordinates.size() == 4);
  Coordinates to{from_coordinates};
  if (type == PermuteType::COPY)
    return to;

  if (type == PermuteType::NHWC_TO_NCHW)
  {
    to.set(0, from_coordinates[0]);
    to.set(1, from_coordinates[3]);
    to.set(2, from_coordinates[1]);
    to.set(3, from_coordinates[2]);
  }
  else
  {
    assert(type == PermuteType::NCHW_TO_NHWC);
    to.set(0, from_coordinates[0]);
    to.set(1, from_coordinates[2]);
    to.set(2, from_coordinates[3]);
    to.set(3, from_coordinates[1]);
  }

  return to;
}

} // namespace ir
} // namespace onert
