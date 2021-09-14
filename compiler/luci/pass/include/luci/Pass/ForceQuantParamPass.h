/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LUCI_FORCE_QUANT_PARAM_PASS_H__
#define __LUCI_FORCE_QUANT_PARAM_PASS_H__

#include <loco.h>

#include <logo/Pass.h>

using TensorVector = std::vector<std::string>;
using ScaleVector = std::vector<float>;
using ZPVector = std::vector<int64_t>;

namespace luci
{

/**
 * @brief Pass to write quantparam (scale, zerop) to the specified tensors
 */
class ForceQuantParamPass : public logo::Pass
{
public:
  ForceQuantParamPass(TensorVector &tensors, ScaleVector &scales, ZPVector &zerops)
    : _tensors{tensors}, _scales{scales}, _zerops{zerops}
  {
    // DO NOTHING
  }
  virtual const char *name(void) const { return "luci::ForceQuantParamPass"; }

public:
  bool run(loco::Graph *graph);

private:
  TensorVector _tensors;
  ScaleVector _scales;
  ZPVector _zerops;
};

} // namespace luci

#endif //__LUCI_FORCE_QUANT_PARAM_PASS_H__
