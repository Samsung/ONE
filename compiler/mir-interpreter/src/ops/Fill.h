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

#ifndef _NNC_CORE_BACKEND_INTERPRETER_FILL_
#define _NNC_CORE_BACKEND_INTERPRETER_FILL_

#include "Common.h"

#include "mir/ShapeRange.h"
#include "mir/Tensor.h"

namespace mir_interpreter
{

template <typename T> struct FillImpl
{
  template <typename F> static void run(mir::TensorVariant &res, F f)
  {
    mir::Tensor<T> res_accessor(res);

    for (const auto &index : mir::ShapeRange(res.getShape()))
    {
      res_accessor.at(index) = f(index);
    }
  }
};

template <typename F> void Fill(mir::TensorVariant &t, F f)
{
  dispatch<FillImpl>(t.getElementType(), t, f);
}

} // namespace mir_interpreter

#endif //_NNC_CORE_BACKEND_INTERPRETER_FILL_
