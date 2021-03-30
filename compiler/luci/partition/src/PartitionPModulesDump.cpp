/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "PartitionPModulesDump.h"

#include "luci/LogHelper.h"

#include <iostream>

namespace luci
{

void dump(std::ostream &os, const PartedModule *pmodule)
{
  os << "--- PartedModule: " << pmodule->group << std::endl;
  os << luci::fmt(pmodule->module->graph());
}

void dump(std::ostream &os, const PartedModules *pmodules)
{
  for (auto &pmodule : pmodules->pmodules)
  {
    dump(os, &pmodule);
  }
  os << std::endl;
}

} // namespace luci

std::ostream &operator<<(std::ostream &os, const luci::PartedModules *pmodules)
{
  luci::dump(os, pmodules);
  return os;
}
