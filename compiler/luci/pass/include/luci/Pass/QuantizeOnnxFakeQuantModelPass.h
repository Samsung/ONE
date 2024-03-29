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

#ifndef __LUCI_QUANTIZE_ONNX_FAKE_QUANT_MODEL_PASS_H__
#define __LUCI_QUANTIZE_ONNX_FAKE_QUANT_MODEL_PASS_H__

#include <loco.h>

#include <logo/Pass.h>

#include <memory>

namespace luci
{

/**
 * @brief Pass to create a quantized graph from a graph fake-quantized on onnx
 */
class QuantizeOnnxFakeQuantModelPass : public logo::Pass
{
public:
  struct Context
  {
    loco::DataType default_activation_dtype = loco::DataType::Unknown;
  };

public:
  QuantizeOnnxFakeQuantModelPass(std::unique_ptr<Context> &&ctx) : _ctx{std::move(ctx)}
  {
    assert(_ctx);                           // FIX_CALLER_UNLESS
    assert(_ctx->default_activation_dtype); // FIX_CALLER_UNLESS
  }

  virtual const char *name(void) const { return "luci::QuantizeOnnxFakeQuantModelPass"; }

public:
  bool run(loco::Graph *graph);

private:
  std::unique_ptr<Context> _ctx;
};

} // namespace luci

#endif //__LUCI_QUANTIZE_ONNX_FAKE_QUANT_MODEL_PASS_H__
