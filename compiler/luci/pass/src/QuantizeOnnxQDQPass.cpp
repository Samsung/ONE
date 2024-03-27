/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "QuantizeOnnxQDQPass.h"
#include "QuantizationUtils.h"

#include <luci/IR/CircleNodes.h>
#include <luci/Profile/CircleNodeOrigin.h>

namespace
{

using namespace luci;

struct OnnxQDQPattern final
{
public:
  OnnxQDQPattern(luci::CircleCustomOut *candidate) { dq_out = candidate; }

public:
  bool matched()
  {
    if (not dq_out)
      return false;

    dq = loco::must_cast<luci::CircleCustom *>(dq_out->input());
    if (not is_onnx_dequantize_linear(dq))
      return false;

    q_out = dynamic_cast<luci::CircleCustomOut *>(dq->inputs(0));
    if (not q_out)
      return false;

    dq_scale = dynamic_cast<luci::CircleConst *>(dq->inputs(1));
    if (not dq_scale)
      return false;

    dq_zerop = dynamic_cast<luci::CircleConst *>(dq->inputs(2));
    if (not dq_zerop)
      return false;

    q = loco::must_cast<luci::CircleCustom *>(q_out->input());
    if (not is_onnx_quantize_linear(q))
      return false;

    input = loco::must_cast<luci::CircleNode *>(q->inputs(0));
    if (input->dtype() != loco::DataType::FLOAT32)
      return false;

    q_scale = dynamic_cast<luci::CircleConst *>(q->inputs(1));
    if (not q_scale)
      return false;

    q_zerop = dynamic_cast<luci::CircleConst *>(q->inputs(2));
    if (not q_zerop)
      return false;

    const auto q_dtype = q->dtype();
    const auto q_scale_dtype = q_scale->dtype();
    const auto q_zerop_dtype = q_zerop->dtype();
    const auto dq_scale_dtype = dq_scale->dtype();
    const auto dq_zerop_dtype = dq_zerop->dtype();

    if (q_scale_dtype != loco::DataType::FLOAT32)
      return false;

    if (dq_scale_dtype != loco::DataType::FLOAT32)
      return false;

    // Invariant from onnx Quantize operator
    if (q_dtype != q_zerop_dtype)
      return false;

    // Invariant from onnx Dequantize operator
    if (q_dtype != dq_zerop_dtype)
      return false;

    // Check length of scale, zp = 1
    if (q_scale->size<loco::DataType::FLOAT32>() != 1)
      return false;

    if (dq_scale->size<loco::DataType::FLOAT32>() != 1)
      return false;

    auto q_zerop_size = 0;
    auto dq_zerop_size = 0;
    switch (q_zerop_dtype)
    {
      case loco::DataType::S16:
        q_zerop_size = q_zerop->size<loco::DataType::S16>();
        dq_zerop_size = dq_zerop->size<loco::DataType::S16>();
        break;
      default:
        throw std::runtime_error("Unsupported zerop dtype in " + q_zerop->name());
    }

    if (q_zerop_size != 1)
      return false;

    if (dq_zerop_size != 1)
      return false;

    return true;
  }

public:
  luci::CircleCustomOut *dq_out = nullptr;
  luci::CircleCustom *dq = nullptr;
  luci::CircleConst *dq_scale = nullptr;
  luci::CircleConst *dq_zerop = nullptr;
  luci::CircleCustomOut *q_out = nullptr;
  luci::CircleCustom *q = nullptr;
  luci::CircleConst *q_scale = nullptr;
  luci::CircleConst *q_zerop = nullptr;
  luci::CircleNode *input = nullptr;
};

class QuantizeOnnxQDQ final
{
public:
  QuantizeOnnxQDQ(const OnnxQDQPattern &p) : _p(p) {}

public:
  void apply(void)
  {
    const auto quantized_dtype = _p.q->dtype();

    // Get scale
    assert(_p.q_scale->dtype() == loco::DataType::FLOAT32);   // FIX_CALLER_UNLESS
    assert(_p.q_scale->size<loco::DataType::FLOAT32>() == 1); // FIX_CALLER_UNLESS
    const float q_scale = _p.q_scale->at<loco::DataType::FLOAT32>(0);

    assert(_p.dq_scale->dtype() == loco::DataType::FLOAT32);   // FIX_CALLER_UNLESS
    assert(_p.dq_scale->size<loco::DataType::FLOAT32>() == 1); // FIX_CALLER_UNLESS
    const float dq_scale = _p.dq_scale->at<loco::DataType::FLOAT32>(0);

    if (q_scale != dq_scale)
      throw std::runtime_error("Invalid scale value in " + _p.dq_scale->name());

    // Get zerop
    int64_t q_zerop = 0;
    int64_t dq_zerop = 0;
    switch (quantized_dtype)
    {
      case loco::DataType::S16:
        assert(_p.q_zerop->size<loco::DataType::S16>() == 1);  // FIX_CALLER_UNLESS
        assert(_p.dq_zerop->size<loco::DataType::S16>() == 1); // FIX_CALLER_UNLESS
        q_zerop = _p.q_zerop->at<loco::DataType::S16>(0);
        dq_zerop = _p.dq_zerop->at<loco::DataType::S16>(0);
        break;
      default:
        throw std::runtime_error("Unsupported zerop dtype in " + _p.q_zerop->name());
    }

    if (q_zerop != dq_zerop)
      throw std::runtime_error("Invalid zerop value in " + _p.dq_zerop->name());

    auto qparam = std::make_unique<luci::CircleQuantParam>();
    {
      qparam->scale.push_back(q_scale);
      qparam->zerop.push_back(q_zerop);
      qparam->quantized_dimension = 0;
    }

    // NOTE We overwrite dtype and qparam to _p.input
    // This can be problematic if a single tensor has
    // multiple different qparams. Let's fix later.
    _p.input->dtype(quantized_dtype);
    _p.input->quantparam(std::move(qparam));

    replace(_p.dq_out).with(_p.input);
  }

private:
  const OnnxQDQPattern &_p;
};

} // namespace

namespace luci
{

/**
 *
 * Quantize pattern
 *
 * [Before]
 *
 *        [CircleNode(fp32)]
 *                |
 *   [CircleCustom(OnnxQuantizeLinear)]
 *                |
 *   [CircleCustom(OnnxDequantizeLinear)]
 *                |
 *           [CircleNode]
 *
 * [After]
 *
 *        [CircleNode(quantized)]
 *                |
 *           [CircleNode]
 */
bool QuantizeOnnxQDQPass::run(loco::Graph *g)
{
  bool changed = false;

  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    if (auto circle_custom_out = dynamic_cast<luci::CircleCustomOut *>(node))
    {
      OnnxQDQPattern p(circle_custom_out);
      if (p.matched())
      {
        QuantizeOnnxQDQ quantize(p);
        quantize.apply();
        changed = true;
      }
    }
  }

  return changed;
}

} // namespace luci
