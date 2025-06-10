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

#ifndef __LUCI_PASS_MODULE_PASS_H__
#define __LUCI_PASS_MODULE_PASS_H__

#include <loco.h>
#include <logo/Pass.h>

#include <luci/IR/Module.h>

#include <stdexcept>

namespace luci
{

class ModulePass
{
public:
  // Run module pass and return false if there was nothing changed
  virtual bool run(luci::Module *) = 0;
};

class Pass : public logo::Pass, public ModulePass
{
public:
  // NOTE adding dummy run() to make compiler happy with "-Werror=overloaded-virtual="
  // clang-format off
  bool run(loco::Graph *) override { throw std::runtime_error("Must inherit"); }
  bool run(luci::Module *) override { throw std::runtime_error("Must inherit"); }
  // clang-format on
};

} // namespace luci

#endif // __LUCI_PASS_MODULE_PASS_H__
