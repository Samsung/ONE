/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __CIRCLE_WEIGHT_DIVIDER_WEIGHT_DIVIDER_H__
#define __CIRCLE_WEIGHT_DIVIDER_WEIGHT_DIVIDER_H__

#include <luci/ImporterEx.h>
#include <luci/CircleOptimizer.h>
#include <luci/DynamicBatchToSingleBatch.h>
#include <luci/Service/ChangeOutputs.h>
#include <luci/Service/Validate.h>
#include <luci/CircleExporter.h>
#include <luci/CircleFileExpContract.h>
#include <luci/UserSettings.h>
#include <luci/IR/CircleNodes.h>

#include <oops/InternalExn.h>
#include <arser/arser.h>
#include <vconone/vconone.h>

#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <string>
#include <fstream>
#include <iostream>

namespace luci
{

class WeightDivider
{
public:
  explicit WeightDivider(luci::Module *module, const std::vector<uint32_t> &ids)
    : _module(module), _ids(ids)
  {
  }

  bool divide();

private:
  luci::Module *_module;
  const std::vector<uint32_t> &_ids;
};

} // namespace luci

#endif // __CIRCLE_WEIGHT_DIVIDER_WEIGHT_DIVIDER_H__
