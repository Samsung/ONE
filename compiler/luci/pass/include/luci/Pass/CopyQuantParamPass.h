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

#ifndef __LUCI_COPY_QUANT_PARAM_PASS_H__
#define __LUCI_COPY_QUANT_PARAM_PASS_H__

#include <loco.h>

#include <logo/Pass.h>

namespace luci
{

/**
 * @brief Pass to copy quantparam (scale, zerop) of a tensor to another tensor
 */
class CopyQuantParamPass : public logo::Pass
{
public:
  using TensorVector = std::vector<std::string>;

public:
  CopyQuantParamPass(TensorVector &src_tensors, TensorVector &dst_tensors)
    : _src_tensors{src_tensors}, _dst_tensors{dst_tensors}
  {
    // DO NOTHING
  }
  virtual const char *name(void) const { return "luci::CopyQuantParamPass"; }

public:
  bool run(loco::Graph *graph);

private:
  TensorVector _src_tensors;
  TensorVector _dst_tensors;
};

} // namespace luci

#endif //__LUCI_COPY_QUANT_PARAM_PASS_H__
