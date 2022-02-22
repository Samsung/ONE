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

#ifndef __LUCI_PROPAGATE_QUANT_PARAM_PASS_H__
#define __LUCI_PROPAGATE_QUANT_PARAM_PASS_H__

#include <logo/Pass.h>

namespace luci
{

/**
 * @brief  Class to propagate quantization parameters of an operator's output to input
 */
struct PropagateQuantParamPass final : public logo::Pass
{
  PropagateQuantParamPass(bool TF_style_maxpool) : _TF_style_maxpool(TF_style_maxpool) {}

  PropagateQuantParamPass() {}

  const char *name(void) const final { return "luci::PropagateQuantParamPass"; }

  bool run(loco::Graph *g) final;

private:
  bool _TF_style_maxpool = false;
};

} // namespace luci

#endif // __LUCI_PROPAGATE_QUANT_PARAM_PASS_H__
