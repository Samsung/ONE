/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LUCI_UNREPRESENTABLE_WARNING_ISSUER_PASS_H__
#define __LUCI_UNREPRESENTABLE_WARNING_ISSUER_PASS_H__

#include <logo/Pass.h>

#include <utility>
#include <luci/IR/CircleNode.h>
#include <helpers/LayerInfoMap.h>
#include "QuantizationParameters.h"

namespace luci
{

/**
 * @brief  Class to convert a quantized model to a fake-quantized fp32 model.
 */
struct UnrepresentableWarningIssuerPass final : public logo::Pass
{
  struct Context
  {
    loco::DataType input_model_dtype = loco::DataType::Unknown;
    loco::DataType output_model_dtype = loco::DataType::Unknown;
    QuantizationGranularity granularity = QuantizationGranularity::ChannelWise;
    std::vector<LayerInfo> layers_info;
  };

  std::unique_ptr<Context> _ctx;
  explicit UnrepresentableWarningIssuerPass(std::unique_ptr<Context> c) : _ctx(std::move(c)) {}

  const char *name(void) const final { return "luci::UnrepresentableWarningIssuerPass"; }

  bool run(loco::Graph *g) final;

protected:

  // For testing
  bool is_unrepr(luci::CircleConst* pConst,
                 loco::DataType input_t,loco::DataType output_t,QuantizationGranularity qgran);

private:
  void warn_if_unrepr(CircleConst *, loco::Graph *);

  LayerInfoMap info_by_name;

  loco::DataType quantize_dtype(const luci::CircleNode *node) {
    auto iter = info_by_name.find(node->name());

    // Return designated quantization dtype
    if (iter != info_by_name.end())
      return iter->second.dtype;

    // Return default quantization dtype
    return _ctx->output_model_dtype;
  };

  QuantizationGranularity quantize_granularity(const luci::CircleNode *node) {
    auto iter = info_by_name.find(node->name());

    // Return designated quantization granularity
    if (iter != info_by_name.end())
      return iter->second.granularity;

    // Return default quantization granularity
    return _ctx->granularity;
  };

};

} // namespace luci

#endif // __LUCI_UNREPRESENTABLE_WARNING_ISSUER_H__
