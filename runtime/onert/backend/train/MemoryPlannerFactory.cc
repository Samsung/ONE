/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "MemoryPlannerFactory.h"

#include "DisposableTensorIndex.h"
#include "ExtraTensorIndex.h"
namespace onert
{
namespace backend
{
namespace train
{

template <typename Index> MemoryPlannerFactory<Index> &MemoryPlannerFactory<Index>::get()
{
  static MemoryPlannerFactory<Index> instance;
  return instance;
}

template <typename Index>
basic::IMemoryPlanner<Index> *MemoryPlannerFactory<Index>::create(const std::string &key)
{
  if (key == "FirstFit")
  {
    return new FirstFitPlanner<Index>();
  }
  else if (key == "Bump")
  {
    return new BumpPlanner<Index>();
  }
  else if (key == "WIC")
  {
    return new WICPlanner<Index>();
  }
  return new FirstFitPlanner<Index>(); // Default Planner
}

// is this necessary?
/**
/usr/bin/ld: libbackend_train.so: undefined reference to
`onert::backend::train::MemoryPlannerFactory<onert::backend::train::DisposableTensorIndex>::create(std::__cxx11::basic_string<char,
std::char_traits<char>, std::allocator<char> > const&)' /usr/bin/ld: libbackend_train.so: undefined
reference to
`onert::backend::train::MemoryPlannerFactory<onert::backend::train::ExtraTensorIndex>::create(std::__cxx11::basic_string<char,
std::char_traits<char>, std::allocator<char> > const&)' /usr/bin/ld: libbackend_train.so: undefined
reference to
`onert::backend::train::MemoryPlannerFactory<onert::backend::train::DisposableTensorIndex>::get()'
/usr/bin/ld: libbackend_train.so: undefined reference to
`onert::backend::train::MemoryPlannerFactory<onert::backend::train::ExtraTensorIndex>::get()'
collect2: error: ld returned 1 exit status
 */
template class MemoryPlannerFactory<DisposableTensorIndex>;
template class MemoryPlannerFactory<ExtraTensorIndex>;

} // namespace train
} // namespace backend
} // namespace onert
