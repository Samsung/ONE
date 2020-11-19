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

#ifndef __LUCI_PASS_H__
#define __LUCI_PASS_H__

#include <loco.h>
#include <logo/Pass.h>

#include <luci/IR/Module.h>

namespace luci
{

class Pass : public logo::Pass
{
public:
  /**
   * @brief  Run the pass
   *
   * @return false if there was nothing changed.
   */

  // Run module pass and return false if there was nothing changed
  virtual bool run(luci::Module *module) = 0;

  // Run graph pass and return false if there was nothing changed
  virtual bool run(loco::Graph *graph) { return false; }
};

} // namespace luci

#endif // __LUCI_PASS_H__
