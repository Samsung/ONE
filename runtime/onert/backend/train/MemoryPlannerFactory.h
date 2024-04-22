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

namespace onert
{
namespace backend
{
namespace train
{

class MemoryPlannerFactory
{
public:
  static MemoryPlannerFactory &get();

private:
  MemoryPlannerFactory() = default;

public:
  template <typename Index> basic::IMemoryPlanner<Index> *create(const std::string &key);
};

template <typename Index>
inline basic::IMemoryPlanner<Index> *MemoryPlannerFactory::create(const std::string &key)
{
  if (key == "FirstFit")
  {
    return new FirstFitPlanner<Index>;
  }
  else if (key == "Bump")
  {
    return new BumpPlanner<Index>;
  }
  else if (key == "WIC")
  {
    return new WICPlanner<Index>;
  }
  return new FirstFitPlanner<Index>; // Default Planner
}

} // namespace train
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRAIN_MEMORY_PLANNER_FACTORY_H__
