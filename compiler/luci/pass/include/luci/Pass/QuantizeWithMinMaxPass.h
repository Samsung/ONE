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

#include <vector>

namespace luci
{

/**
 * @brief Pass to quantize activation, weights, and bias
 */
class QuantizeWithMinMaxPass : public logo::Pass
{
public:
  struct Context
  {
    loco::DataType input_model_dtype = loco::DataType::Unknown;
    loco::DataType output_model_dtype = loco::DataType::Unknown;
    QuantizationGranularity granularity = QuantizationGranularity::ChannelWise;
    std::vector<loco::DataType> input_types;
    std::vector<loco::DataType> output_types;
    bool TF_style_maxpool = false;
    bool save_min_max = false;
    std::vector<LayerInfo> layers_info;
  };

public:
  QuantizeWithMinMaxPass(std::unique_ptr<Context> &&ctx) : _ctx{std::move(ctx)}
  {
    // DO NOTHING
  }

  virtual const char *name(void) const { return "luci::QuantizeWithMinMaxPass"; }

public:
  bool run(loco::Graph *graph);

private:
  void set_input_type(loco::Graph *graph) const;
  void set_output_type(loco::Graph *graph) const;

private:
  std::unique_ptr<Context> _ctx;
};

} // namespace luci

#endif //__LUCI_QUANTIZE_WITH_MINMAX_PASS_H__
