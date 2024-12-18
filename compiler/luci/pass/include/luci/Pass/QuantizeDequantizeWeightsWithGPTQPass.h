/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LUCI_QUANTIZE_WEIGHTS_WITH_GPTQ_PASS_H__
#define __LUCI_QUANTIZE_WEIGHTS_WITH_GPTQ_PASS_H__

#include <luci/Pass/QuantizationParameters.h>
#include <luci/IR/CircleNode.h>

#include <logo/Pass.h>
#include <loco.h>

#include <unordered_map>

namespace luci
{

using HessianMap = std::unordered_map<const luci::CircleNode *, std::vector<float>>;

/**
 * @brief Pass to quantize weights with GPTQ algorithm
 */
class QuantizeDequantizeWeightsWithGPTQPass : public logo::Pass
{
public:
  struct Context
  {
    loco::DataType input_model_dtype = loco::DataType::Unknown;
    loco::DataType output_model_dtype = loco::DataType::Unknown;
    QuantizationGranularity granularity = QuantizationGranularity::ChannelWise;
    std::vector<LayerInfo> layers_info;
  };

public:
  QuantizeDequantizeWeightsWithGPTQPass(std::unique_ptr<Context> &&ctx, HessianMap *hessian_map)
    : _ctx{std::move(ctx)}, _hessian_map{hessian_map}
  {
    // DO NOTHING
  }
  virtual const char *name(void) const { return "luci::QuantizeDequantizeWeightsWithGPTQPass"; }

public:
  bool run(loco::Graph *graph);

private:
  std::unique_ptr<Context> _ctx;
  HessianMap *_hessian_map = nullptr;
};

} // namespace luci

#endif //__LUCI_QUANTIZE_WEIGHTS_WITH_GPTQ_PASS_H__
