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

#ifndef __LUCI_QUANTIZE_WITH_MINMAX_PASS_H__
#define __LUCI_QUANTIZE_WITH_MINMAX_PASS_H__

#include <loco.h>

#include <logo/Pass.h>

#include <luci/Pass/QuantizationParameters.h>

namespace luci
{

/**
 * @brief Pass to quantize activation, weights, and bias
 */
class QuantizeWithMinMaxPass : public logo::Pass
{
public:
  QuantizeWithMinMaxPass(loco::DataType input_dtype, loco::DataType output_dtype,
                         QuantizationGranularity granularity)
    : _input_dtype{input_dtype}, _output_dtype{output_dtype}, _granularity{granularity}
  {
    // DO NOTHING
  }
  virtual const char *name(void) const { return "luci::QuantizeWithMinMaxPass"; }

public:
  bool run(loco::Graph *graph);

private:
  loco::DataType _input_dtype;
  loco::DataType _output_dtype;
  QuantizationGranularity _granularity;
};

} // namespace luci

#endif //__LUCI_QUANTIZE_WITH_MINMAX_PASS_H__
