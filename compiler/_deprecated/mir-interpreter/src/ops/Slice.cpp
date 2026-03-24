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

#include "Slice.h"

#include "Fill.h"
#include "Common.h"

#include "mir/Tensor.h"
#include "mir/ShapeRange.h"

namespace mir_interpreter
{

template <typename T> struct SliceImpl
{
  static void run(const mir::TensorVariant &arg, const mir::Shape &starts, mir::TensorVariant &res);
};

template <typename T>
void SliceImpl<T>::run(const mir::TensorVariant &arg, const mir::Shape &starts,
                       mir::TensorVariant &res)
{
  mir::Tensor<T> input(arg);
  mir::Tensor<T> output(res);

  for (auto id : mir::ShapeRange(res.getShape()))
  {
    mir::Index idx = mir_interpreter::shift(id, starts);
    output.at(id) = input.at(idx);
  }
}

void Slice(const mir::TensorVariant &arg, const mir::Shape &starts, mir::TensorVariant &res)
{
  dispatch<SliceImpl>(arg.getElementType(), arg, starts, res);
}

} // namespace mir_interpreter
