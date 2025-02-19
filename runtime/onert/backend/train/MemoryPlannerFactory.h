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

#ifndef __ONERT_BACKEND_TRAIN_MEMORY_PLANNER_FACTORY_H__
#define __ONERT_BACKEND_TRAIN_MEMORY_PLANNER_FACTORY_H__

#include "MemoryPlanner.h"

#include <string>

namespace onert::backend::train
{

template <typename Index> class MemoryPlannerFactory
{
public:
  static MemoryPlannerFactory<Index> &get();

private:
  MemoryPlannerFactory() = default;

public:
  // Currently, only the memory planner for DisposableTensor is supported
  basic::IMemoryPlanner<Index> *create(std::string_view key);
};

} // namespace onert::backend::train

#endif // __ONERT_BACKEND_TRAIN_MEMORY_PLANNER_FACTORY_H__
