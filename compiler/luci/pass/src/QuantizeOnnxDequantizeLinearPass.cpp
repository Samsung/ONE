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

#include "QuantizeOnnxDequantizeLinearPass.h"
#include "QuantizationUtils.h"

#include <luci/IR/CircleNodes.h>
#include <luci/Profile/CircleNodeOrigin.h>

#include <flatbuffers/flexbuffers.h>

namespace
{

using namespace luci;

// Return true if all values of node are within value_range
// value_range: [min, max]
template <loco::DataType DT>
bool value_range(const luci::CircleConst *node, const std::pair<int64_t, int64_t> &value_range)
{
  const auto min = value_range.first;
  const auto max = value_range.second;

  auto size = node->size<DT>();
  for (uint32_t i = 0; i < size; i++)
  {
    const auto val = static_cast<int64_t>(node->at<DT>(i));
    if (val < min or val > max)
      return false;
  }

  return true;
}

std::vector<float> get_scales(const luci::CircleConst *node)
{
  assert(node); // FIX_CALLER_UNLESS

  const auto num_scales = node->size<loco::DataType::FLOAT32>();
  std::vector<float> scales(num_scales);
  for (uint32_t i = 0; i < num_scales; ++i)
  {
    scales[i] = node->at<loco::DataType::FLOAT32>(i);
  }

  return scales;
}

template <loco::DataType DT> std::vector<int64_t> get_zerops(const luci::CircleConst *node)
{
  assert(node); // FIX_CALLER_UNLESS

  const auto num_zerops = node->size<DT>();
  std::vector<int64_t> zerops(num_zerops);
  for (uint32_t i = 0; i < num_zerops; ++i)
  {
    zerops[i] = node->at<DT>(i);
  }

  return zerops;
}

int32_t get_axis(const luci::CircleCustom *node)
{
  assert(node); // FIX_CALLER_UNLESS

  const auto custom_options = node->custom_options();
  const auto map = flexbuffers::GetRoot(custom_options).AsMap();

  return map["axis"].IsNull() ? 0 : map["axis"].AsInt32();
}

class OnnxDequantizeLinearPattern final
{
public:
  OnnxDequantizeLinearPattern(luci::CircleCustomOut *candidate) { custom_out = candidate; }

public:
  bool matched()
  {
    if (not custom_out)
      return false;

    dequantize = loco::must_cast<luci::CircleCustom *>(custom_out->input());
    if (not is_onnx_dequantize_linear(dequantize))
      return false;

    input = dynamic_cast<luci::CircleConst *>(dequantize->inputs(0));
    if (not input)
      return false;

    scale = dynamic_cast<luci::CircleConst *>(dequantize->inputs(1));
    if (not scale)
      return false;

    zerop = dynamic_cast<luci::CircleConst *>(dequantize->inputs(2));
    if (not zerop)
      return false;

    const auto input_dtype = input->dtype();
    const auto scale_dtype = scale->dtype();
    const auto zerop_dtype = zerop->dtype();

    if (scale_dtype != loco::DataType::FLOAT32)
      return false;

    // Invariant from onnx DequantizeLinear operator
    if (input_dtype != zerop_dtype)
      return false;

    return true;
  }

public:
  luci::CircleCustomOut *custom_out = nullptr;
  luci::CircleCustom *dequantize = nullptr;
  luci::CircleConst *input = nullptr;
  luci::CircleConst *scale = nullptr;
  luci::CircleConst *zerop = nullptr;
};

// Temporary class for our in-house model
// This is for per-tensor quantized LN const
// uint8 weight, int16 zerop, fp32 scale
// NOTE weight dtype != zerop dtype breaks invariant of
// onnx DequantizeLinear. That's why this class is a hack.
class OnnxDequantizeLinearPatternV2 final
{
public:
  OnnxDequantizeLinearPatternV2(luci::CircleCustomOut *candidate) { custom_out = candidate; }

public:
  bool matched()
  {
    if (not custom_out)
      return false;

    dequantize = loco::must_cast<luci::CircleCustom *>(custom_out->input());
    if (not is_onnx_dequantize_linear(dequantize))
      return false;

    input = dynamic_cast<luci::CircleConst *>(dequantize->inputs(0));
    if (not input)
      return false;

    scale = dynamic_cast<luci::CircleConst *>(dequantize->inputs(1));
    if (not scale)
      return false;

    zerop = dynamic_cast<luci::CircleConst *>(dequantize->inputs(2));
    if (not zerop)
      return false;

    const auto input_dtype = input->dtype();
    const auto scale_dtype = scale->dtype();
    const auto zerop_dtype = zerop->dtype();

    if (scale_dtype != loco::DataType::FLOAT32)
      return false;

    if (input_dtype != loco::DataType::U8)
      return false;

    if (zerop_dtype != loco::DataType::S16)
      return false;

    return true;
  }

public:
  luci::CircleCustomOut *custom_out = nullptr;
  luci::CircleCustom *dequantize = nullptr;
  luci::CircleConst *input = nullptr;
  luci::CircleConst *scale = nullptr;
  luci::CircleConst *zerop = nullptr;
};

class QuantizeOnnxDequantizeLinear final
{
public:
  QuantizeOnnxDequantizeLinear(const OnnxDequantizeLinearPattern &p) : _p(p) {}

public:
  void apply(void)
  {
    // The final const's dtype is the same with input_dtype by default
    auto const_dtype = _p.input->dtype();
    if (const_dtype == loco::DataType::U8)
    {
      // Onnx does not support int4/uint4 as of writing. We assume uint8
      // tensor is quantized in int4/uint4 if values are within [0,15]
      if (value_range<loco::DataType::U8>(_p.input, {0, 15}))
      {
        if (value_range<loco::DataType::U8>(_p.zerop, {8, 8}))
        {
          const_dtype = loco::DataType::S4;
        }
        else if (value_range<loco::DataType::U8>(_p.zerop, {0, 15}))
        {
          const_dtype = loco::DataType::U4;
        }
      }
    }

    luci::CircleConst *quant_const = nullptr;
    switch (const_dtype)
    {
      case loco::DataType::S4:
        quant_const = gen_s4_quant();
        break;
      case loco::DataType::U4:
        quant_const = gen_u4_quant();
        break;
      case loco::DataType::U8:
        quant_const = gen_u8_quant();
        break;
      default:
        throw std::runtime_error("Unsupported quantized dtype");
    }

    assert(quant_const); // FIX_ME_UNLESS

    // set origin
    std::vector<std::shared_ptr<luci::CircleNodeOrigin>> origin_vec{
      luci::get_origin(_p.dequantize), luci::get_origin(_p.input), luci::get_origin(_p.scale),
      luci::get_origin(_p.zerop)};

    luci::add_origin(quant_const, luci::composite_origin(origin_vec));

    replace(_p.custom_out).with(quant_const);
  }

private:
  luci::CircleConst *gen_s4_quant(void)
  {
    assert(_p.input->dtype() == loco::DataType::U8);      // FIX_CALLER_UNLESS
    assert(_p.scale->dtype() == loco::DataType::FLOAT32); // FIX_CALLER_UNLESS
    assert(_p.zerop->dtype() == loco::DataType::U8);      // FIX_CALLER_UNLESS

    auto quantized_node = _p.dequantize->graph()->nodes()->create<luci::CircleConst>();
    quantized_node->dtype(loco::DataType::S4);
    quantized_node->rank(_p.input->rank());
    for (uint32_t i = 0; i < _p.input->rank(); ++i)
    {
      quantized_node->dim(i) = _p.input->dim(i);
    }
    quantized_node->shape_status(luci::ShapeStatus::VALID);

    // Create S4 CircleConst
    // NOTE S4 is saved as S8 in luci::CircleConst
    const auto num_elems = _p.input->size<loco::DataType::U8>();
    quantized_node->size<loco::DataType::S4>(num_elems);
    for (uint32_t i = 0; i < num_elems; i++)
    {
      const uint8_t u8_val = _p.input->at<loco::DataType::U8>(i);
      assert(u8_val <= 15); // FIX_CALLER_UNLESS
      quantized_node->at<loco::DataType::S4>(i) = static_cast<int8_t>(u8_val) - 8;
    }

    auto qparam = std::make_unique<luci::CircleQuantParam>();
    {
      const std::vector<float> scale_vector = get_scales(_p.scale);
      const std::vector<int64_t> zerop_vector = get_zerops<loco::DataType::U8>(_p.zerop);

      if (scale_vector.size() != zerop_vector.size())
        throw std::runtime_error("Scale/Zerop size mismatches in " + _p.dequantize->name());

      const int32_t qdim = get_axis(_p.dequantize);

      qparam->scale = scale_vector;
      qparam->zerop = zerop_vector;
      qparam->quantized_dimension = qdim;
    }

    quantized_node->quantparam(std::move(qparam));

    quantized_node->name(_p.input->name());

    return quantized_node;
  }

  luci::CircleConst *gen_u4_quant(void)
  {
    assert(_p.input->dtype() == loco::DataType::U8);      // FIX_CALLER_UNLESS
    assert(_p.scale->dtype() == loco::DataType::FLOAT32); // FIX_CALLER_UNLESS
    assert(_p.zerop->dtype() == loco::DataType::U8);      // FIX_CALLER_UNLESS

    auto quantized_node = _p.dequantize->graph()->nodes()->create<luci::CircleConst>();
    quantized_node->dtype(loco::DataType::U4);
    quantized_node->rank(_p.input->rank());
    for (uint32_t i = 0; i < _p.input->rank(); ++i)
    {
      quantized_node->dim(i) = _p.input->dim(i);
    }
    quantized_node->shape_status(luci::ShapeStatus::VALID);

    // Create U4 CircleConst
    // NOTE U4 is saved as U8 in luci::CircleConst
    const auto num_elems = _p.input->size<loco::DataType::U8>();
    quantized_node->size<loco::DataType::U4>(num_elems);
    for (uint32_t i = 0; i < num_elems; i++)
    {
      const uint8_t u8_val = _p.input->at<loco::DataType::U8>(i);
      assert(u8_val <= 15); // FIX_CALLER_UNLESS
      quantized_node->at<loco::DataType::U4>(i) = u8_val;
    }

    auto qparam = std::make_unique<luci::CircleQuantParam>();
    {
      const std::vector<float> scale_vector = get_scales(_p.scale);
      const std::vector<int64_t> zerop_vector = get_zerops<loco::DataType::U8>(_p.zerop);

      if (scale_vector.size() != zerop_vector.size())
        throw std::runtime_error("Scale/Zerop size mismatches in " + _p.dequantize->name());

      const int32_t qdim = get_axis(_p.dequantize);

      qparam->scale = scale_vector;
      qparam->zerop = zerop_vector;
      qparam->quantized_dimension = qdim;
    }

    quantized_node->quantparam(std::move(qparam));

    quantized_node->name(_p.input->name());

    return quantized_node;
  }

  luci::CircleConst *gen_u8_quant(void)
  {
    assert(_p.input->dtype() == loco::DataType::U8);      // FIX_CALLER_UNLESS
    assert(_p.scale->dtype() == loco::DataType::FLOAT32); // FIX_CALLER_UNLESS
    assert(_p.zerop->dtype() == loco::DataType::U8);      // FIX_CALLER_UNLESS

    auto quantized_node = _p.dequantize->graph()->nodes()->create<luci::CircleConst>();
    quantized_node->dtype(loco::DataType::U8);
    quantized_node->rank(_p.input->rank());
    for (uint32_t i = 0; i < _p.input->rank(); ++i)
    {
      quantized_node->dim(i) = _p.input->dim(i);
    }
    quantized_node->shape_status(luci::ShapeStatus::VALID);

    // Create U8 CircleConst
    const auto num_elems = _p.input->size<loco::DataType::U8>();
    quantized_node->size<loco::DataType::U8>(num_elems);
    for (uint32_t i = 0; i < num_elems; i++)
    {
      const uint8_t u8_val = _p.input->at<loco::DataType::U8>(i);
      quantized_node->at<loco::DataType::U8>(i) = u8_val;
    }

    auto qparam = std::make_unique<luci::CircleQuantParam>();
    {
      const std::vector<float> scale_vector = get_scales(_p.scale);
      const std::vector<int64_t> zerop_vector = get_zerops<loco::DataType::U8>(_p.zerop);

      if (scale_vector.size() != zerop_vector.size())
        throw std::runtime_error("Scale/Zerop size mismatches in " + _p.dequantize->name());

      const int32_t qdim = get_axis(_p.dequantize);

      qparam->scale = scale_vector;
      qparam->zerop = zerop_vector;
      qparam->quantized_dimension = qdim;
    }

    quantized_node->quantparam(std::move(qparam));

    quantized_node->name(_p.input->name());

    return quantized_node;
  }

private:
  const OnnxDequantizeLinearPattern &_p;
};

// Temporary class to handle our in-house model
class QuantizeOnnxDequantizeLinearV2 final
{
public:
  QuantizeOnnxDequantizeLinearV2(const OnnxDequantizeLinearPatternV2 &p) : _p(p) {}

public:
  void apply(void)
  {
    auto const_dtype = _p.zerop->dtype();

    luci::CircleConst *quant_const = nullptr;
    switch (const_dtype)
    {
      case loco::DataType::S16:
        quant_const = gen_s16_quant();
        break;
      default:
        throw std::runtime_error("Unsupported quantized dtype");
    }

    assert(quant_const); // FIX_ME_UNLESS

    // set origin
    std::vector<std::shared_ptr<luci::CircleNodeOrigin>> origin_vec{
      luci::get_origin(_p.dequantize), luci::get_origin(_p.input), luci::get_origin(_p.scale),
      luci::get_origin(_p.zerop)};

    luci::add_origin(quant_const, luci::composite_origin(origin_vec));

    replace(_p.custom_out).with(quant_const);
  }

private:
  luci::CircleConst *gen_s16_quant(void)
  {
    assert(_p.input->dtype() == loco::DataType::U8);      // FIX_CALLER_UNLESS
    assert(_p.scale->dtype() == loco::DataType::FLOAT32); // FIX_CALLER_UNLESS
    assert(_p.zerop->dtype() == loco::DataType::S16);     // FIX_CALLER_UNLESS

    auto quantized_node = _p.dequantize->graph()->nodes()->create<luci::CircleConst>();
    quantized_node->dtype(loco::DataType::S16);
    quantized_node->rank(_p.input->rank());
    for (uint32_t i = 0; i < _p.input->rank(); ++i)
    {
      quantized_node->dim(i) = _p.input->dim(i);
    }
    quantized_node->shape_status(luci::ShapeStatus::VALID);

    // Create S16 CircleConst
    const auto num_elems = _p.input->size<loco::DataType::U8>();
    quantized_node->size<loco::DataType::S16>(num_elems);
    for (uint32_t i = 0; i < num_elems; i++)
    {
      const uint8_t u8_val = _p.input->at<loco::DataType::U8>(i);
      quantized_node->at<loco::DataType::S16>(i) = static_cast<int16_t>(u8_val);
    }

    auto qparam = std::make_unique<luci::CircleQuantParam>();
    {
      const std::vector<float> scale_vector = get_scales(_p.scale);
      const std::vector<int64_t> zerop_vector = get_zerops<loco::DataType::S16>(_p.zerop);

      if (scale_vector.size() != zerop_vector.size())
        throw std::runtime_error("Scale/Zerop size mismatches in " + _p.dequantize->name());

      const int32_t qdim = get_axis(_p.dequantize);

      qparam->scale = scale_vector;
      qparam->zerop = zerop_vector;
      qparam->quantized_dimension = qdim;
    }

    quantized_node->quantparam(std::move(qparam));

    quantized_node->name(_p.input->name());

    return quantized_node;
  }

private:
  const OnnxDequantizeLinearPatternV2 &_p;
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
 *      [CircleConst(quantized)]
 *                |
 *   [CircleCustom(OnnxDequantizeLinear)]
 *                |
 *           [CircleNode]
 *
 * [After]
 *
 *         [CircleConst(quantized)]
 *                |
 *           [CircleNode]
 */
bool QuantizeOnnxDequantizeLinearPass::run(loco::Graph *g)
{
  bool changed = false;

  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    if (auto circle_custom_out = dynamic_cast<luci::CircleCustomOut *>(node))
    {
      OnnxDequantizeLinearPattern p(circle_custom_out);
      if (p.matched())
      {
        QuantizeOnnxDequantizeLinear quantize(p);
        quantize.apply();
        changed = true;
      }

      // TODO Remove V2 classes
      OnnxDequantizeLinearPatternV2 p2(circle_custom_out);
      if (p2.matched())
      {
        QuantizeOnnxDequantizeLinearV2 quantize(p2);
        quantize.apply();
        changed = true;
      }
    }
  }

  return changed;
}

} // namespace luci
