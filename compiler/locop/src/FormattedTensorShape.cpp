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

#include "locop/FormattedTensorShape.h"

namespace loco
{

std::ostream &operator<<(std::ostream &os, const loco::Dimension &d)
{
  os << (d.known() ? std::to_string(d.value()) : std::string{"?"});
  return os;
}

} // namespace loco

namespace locop
{

void FormattedTensorShape<TensorShapeFormat::Plain>::dump(std::ostream &os) const
{
  if (_ptr->rank() > 0)
  {
    os << _ptr->dim(0);

    for (uint32_t axis = 1; axis < _ptr->rank(); ++axis)
    {
      os << " x " << _ptr->dim(axis);
    }
  }
}

} // namespace locop

namespace locop
{

void FormattedTensorShape<TensorShapeFormat::Bracket>::dump(std::ostream &os) const
{
  os << "[";

  if (_ptr->rank() > 0)
  {
    os << " " << _ptr->dim(0);

    for (uint32_t axis = 1; axis < _ptr->rank(); ++axis)
    {
      os << " x " << _ptr->dim(axis);
    }
  }

  os << " ]";
}

} // namespace locop
