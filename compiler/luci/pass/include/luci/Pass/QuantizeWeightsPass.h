/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LUCI_QUANTIZE_WEIGHTS_PASS_H__
#define __LUCI_QUANTIZE_WEIGHTS_PASS_H__

#include <loco.h>

#include <logo/Pass.h>

#include <luci/Pass/QuantizationParameters.h>

namespace luci
{

/**
 * @brief Pass to quantize weights
 */
class QuantizeWeightsPass : public logo::Pass
{
public:
  struct Context
  {
    loco::DataType input_model_dtype = loco::DataType::Unknown;
    loco::DataType output_model_dtype = loco::DataType::Unknown;
    QuantizationGranularity granularity = QuantizationGranularity::ChannelWise;
  };

public:
  QuantizeWeightsPass(std::unique_ptr<Context> &&ctx) : _ctx{std::move(ctx)}
  {
    // DO NOTHING
  }

public:
  QuantizeWeightsPass(loco::DataType input_model_dtype, loco::DataType output_model_dtype,
                      QuantizationGranularity granularity)
  {
    _ctx = std::make_unique<Context>();
    {
      _ctx->input_model_dtype = input_model_dtype;
      _ctx->output_model_dtype = output_model_dtype;
      _ctx->granularity = granularity;
    }
  }
  virtual const char *name(void) const { return "luci::QuantizeWeightsPass"; }

public:
  bool run(loco::Graph *graph);

private:
  std::unique_ptr<Context> _ctx;
};

} // namespace luci

#endif //__LUCI_QUANTIZE_WEIGHTS_PASS_H__
