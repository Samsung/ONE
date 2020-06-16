/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "backend/ITensor.h"

namespace onert
{
namespace backend
{

ir::Shape ITensor::getShape() const
{
  onert::ir::Shape shape(num_dimensions());
  for (uint32_t d = 0; d < num_dimensions(); d++)
    shape.dim(d) = dimension(d);

  return shape;
}

} // namespace backend
} // namespace onert
