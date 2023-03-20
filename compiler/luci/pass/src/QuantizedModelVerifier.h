/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LUCI_QUANTIZED_MODEL_VERIFIER_H__
#define __LUCI_QUANTIZED_MODEL_VERIFIER_H__

#include "luci/Pass/QuantizationParameters.h"

#include <loco.h>

#include <memory>

namespace luci
{

/**
 * @brief  Class to verify quantized model
 *
 * TODO Move this to luci/service
 */
struct QuantizedModelVerifier
{
public:
  struct Context
  {
    loco::DataType output_model_dtype = loco::DataType::Unknown;
    QuantizationGranularity granularity = QuantizationGranularity::ChannelWise;
    std::vector<loco::DataType> input_types;
    std::vector<loco::DataType> output_types;
    bool TF_style_maxpool = false;
    std::vector<LayerInfo> layers_info;
  };

public:
  QuantizedModelVerifier(std::unique_ptr<Context> &&ctx) : _ctx{std::move(ctx)}
  {
    // DO NOTHING
  }

  void verify(loco::Graph *g);

private:
  std::unique_ptr<Context> _ctx;
};

} // namespace luci

#endif // __LUCI_QUANTIZED_MODEL_VERIFIER_H__
