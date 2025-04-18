/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "MemoryPlanner.h"

namespace onert::backend::basic
{

MemoryPlannerFactory &MemoryPlannerFactory::get()
{
  static MemoryPlannerFactory instance;
  return instance;
}

IMemoryPlanner<ir::OperandIndex> *MemoryPlannerFactory::create(const std::string &key)
{
  if (key == "FirstFit")
  {
    return new FirstFitPlanner;
  }
  else if (key == "Bump")
  {
    return new BumpPlanner;
  }
  else if (key == "WIC")
  {
    return new WICPlanner;
  }
  return new FirstFitPlanner; // Default Planner
}

} // namespace onert::backend::basic
