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

#ifndef __MPQSOLVER_CORE_QUANTIZER_H__
#define __MPQSOLVER_CORE_QUANTIZER_H__

#include <luci/IR/Module.h>
#include <luci/CircleQuantizer.h>

#include <string>
#include <vector>
#include <memory>

namespace mpqsolver
{
namespace core
{

using LayerParam = luci::CircleQuantizer::Options::LayerParam;
using LayerParams = std::vector<std::shared_ptr<LayerParam>>;

struct QuantizerHook
{
  /**
   * @brief called on successfull quantization
   * @param module quantized module
   */
  virtual void onQuantized(luci::Module *module) const = 0;
};

class Quantizer
{
public:
  struct Context
  {
    std::string output_model_dtype = "uint8";
    std::string granularity = "channel";
    std::string input_type = "uint8";
    std::string output_type = "uint8";
    bool TF_style_maxpool = false;
    bool save_min_max = false;
    // TODO Support layer info
  };

public:
  Quantizer(const Context &ctx) : _ctx(ctx) {}

  /**
   * @brief set hook on the end of quantization event
   */
  void setHook(const QuantizerHook *callback);

  /**
   * @brief quantize recorded module (min/max initialized) with specified parameters
   * returns true on success
   */
  bool quantize(luci::Module *module, const std::string &quant_dtype, LayerParams &layer_params);

  /**
   * @brief quantize recorded module (min/max initialized) with specified parameters
   * returns true on success
   */
  bool quantize(luci::Module *module, LayerParams &layer_params);

  /**
   * @brief fake_quantize recorded module (min/max initialized) with specified parameters
   * returns true on success
   */
  bool fakeQuantize(luci::Module *module, const std::string &quant_dtype,
                    LayerParams &layer_params);

  const Context &getContext() const { return _ctx; }

private:
  Context _ctx;
  const QuantizerHook *_hook = nullptr;
};

} // namespace core
} // namespace mpqsolver

#endif //__MPQSOLVER_CORE_QUANTIZER_H__
