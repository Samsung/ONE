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

#ifndef _NNC_CORE_BACKEND_INTERPRETER_COMMON_
#define _NNC_CORE_BACKEND_INTERPRETER_COMMON_

#include "mir/Tensor.h"
#include "mir/TensorVariant.h"
#include "mir/DataType.h"
#include "mir/Shape.h"
#include "mir/Index.h"

namespace mir_interpreter
{

template <template <typename> class F, typename... Args>
void dispatch(mir::DataType dt, Args &&...args)
{
  switch (dt)
  {
    case mir::DataType::FLOAT32:
      return F<float>::run(std::forward<Args>(args)...);
    case mir::DataType::FLOAT64:
      return F<double>::run(std::forward<Args>(args)...);
    case mir::DataType::INT32:
      return F<int32_t>::run(std::forward<Args>(args)...);
    case mir::DataType::INT64:
      return F<int64_t>::run(std::forward<Args>(args)...);
    case mir::DataType::UINT8:
      return F<uint8_t>::run(std::forward<Args>(args)...);
    case mir::DataType::UNKNOWN:
      throw std::runtime_error{"Unknown datatype met during operation execution"};
    default:
      throw std::runtime_error{"mir::DataType enum mismatch"};
  }
}

template <typename T> void erase(mir::TensorVariant &tv)
{
  size_t element_count = tv.getShape().numElements();
  for (size_t i = 0; i < element_count; ++i)
  {
    auto ptr = tv.atOffset(i);
    *reinterpret_cast<T *>(ptr) = 0;
  }
}

mir::Index shift(const mir::Index &in_index, const mir::Shape &shift_from);

} // namespace mir_interpreter

#endif // _NNC_CORE_BACKEND_INTERPRETER_COMMON_
