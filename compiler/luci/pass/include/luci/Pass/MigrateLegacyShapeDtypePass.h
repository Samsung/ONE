/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LUCI_MIGRATE_LEGACY_SHAPE_DTYPE_PASS_H__
#define __LUCI_MIGRATE_LEGACY_SHAPE_DTYPE_PASS_H__

#include <loco.h>

#include <luci/ModulePass.h>

namespace luci
{

/**
 * @brief Pass to copy shape/dtype of loco to circle node
 *
 * CAUTION : This pass will be removed after refactoring is finished
 */
class MigrateLegacyShapeDtypePass : public luci::Pass
{
public:
  virtual const char *name(void) const { return "luci::MigrateLegacyShapeDtypePass"; }

public:
  bool run(luci::Module *m);
  bool run(loco::Graph *graph);
};

} // namespace luci

#endif //__LUCI_MIGRATE_LEGACY_SHAPE_DTYPE_PASS_H__
