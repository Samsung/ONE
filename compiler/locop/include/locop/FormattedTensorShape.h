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

#ifndef __LOCOP_FORMATTED_TENSOR_SHAPE_H__
#define __LOCOP_FORMATTED_TENSOR_SHAPE_H__

#include "locop/Interfaces.h"

#include <loco/IR/TensorShape.h>

namespace locop
{

enum class TensorShapeFormat
{
  // D_0 x D_1 x ... D_N
  Plain,
  // [ D_0 x D_1 x D_2 x ... ]
  Bracket,
};

template <TensorShapeFormat Format> class FormattedTensorShape;

template <>
class FormattedTensorShape<TensorShapeFormat::Plain> final : public Spec<Interface::Formatted>
{
public:
  FormattedTensorShape(const loco::TensorShape *ptr) : _ptr{ptr}
  {
    // DO NOTHING
  }

public:
  void dump(std::ostream &os) const final;

private:
  const loco::TensorShape *_ptr = nullptr;
};

template <>
class FormattedTensorShape<TensorShapeFormat::Bracket> final : public Spec<Interface::Formatted>
{
public:
  FormattedTensorShape(const loco::TensorShape *ptr) : _ptr{ptr}
  {
    // DO NOTHING
  }

public:
  void dump(std::ostream &os) const final;

private:
  const loco::TensorShape *_ptr = nullptr;
};

template <TensorShapeFormat F> FormattedTensorShape<F> fmt(loco::TensorShape *ptr)
{
  return FormattedTensorShape<F>{ptr};
}

} // namespace locop

#endif // __LOCOP_FORMATTED_TENSOR_SHAPE_H__
