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

#include "luci/Pass/PropagateQParamBackwardPass.h"
#include "QuantizationUtils.h"

#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleNodeVisitor.h>
#include <luci/Service/Nodes/CircleConst.h>
#include <luci/Log.h>

#include <cmath>
#include <functional>

namespace
{

void quant_const_values(luci::CircleConst *const_node, float scaling_factor, float zerop,
                        loco::DataType quant_type)
{
  uint32_t size = const_node->size<loco::DataType::FLOAT32>();

  const float scaling_factor_inv = 1.0 / scaling_factor;
  std::vector<int32_t> quantized_values(size);
  for (uint32_t i = 0; i < size; ++i)
  {
    auto data = static_cast<double>(const_node->at<loco::DataType::FLOAT32>(i));
    double quantized_float = std::round(data * scaling_factor_inv) + zerop;
    constexpr auto int_max = static_cast<double>(std::numeric_limits<int32_t>::max());
    constexpr auto int_min = static_cast<double>(std::numeric_limits<int32_t>::min());
    quantized_float = std::min(int_max, std::max(int_min, quantized_float));

    quantized_values[i] = static_cast<int32_t>(quantized_float);
  }

  switch (quant_type)
  {
    case loco::DataType::U8:
      const_node->dtype(loco::DataType::U8);      // change the type of tensor
      const_node->size<loco::DataType::U8>(size); // resize tensor
      for (uint32_t i = 0; i < size; ++i)
        const_node->at<loco::DataType::U8>(i) = std::min(255, std::max(0, quantized_values[i]));
      break;
    case loco::DataType::S16:
      assert(zerop == 0);
      const_node->dtype(loco::DataType::S16);      // change the type of tensor
      const_node->size<loco::DataType::S16>(size); // resize tensor
      for (uint32_t i = 0; i < size; ++i)
        const_node->at<loco::DataType::S16>(i) =
          std::min(32767, std::max(-32767, quantized_values[i]));
      break;
    default:
      throw std::runtime_error("Unsupported data type");
  }
}

void quant_const(luci::CircleConst *node, loco::DataType quant_type)
{
  assert(node->dtype() == loco::DataType::FLOAT32);

  float min = std::numeric_limits<float>::max();
  float max = std::numeric_limits<float>::lowest();
  for (uint32_t i = 0; i < node->size<loco::DataType::FLOAT32>(); i++)
  {
    auto data = node->at<loco::DataType::FLOAT32>(i);
    min = data < min ? data : min;
    max = data > max ? data : max;
  }

  float scaling_factor{0.0};
  int64_t zp{0};
  float nudged_min{0.0};
  float nudged_max{0.0};

  switch (quant_type)
  {
    case loco::DataType::U8:
      asymmetric_wquant_with_minmax_per_layer(node, min, max, scaling_factor, zp, nudged_min,
                                              nudged_max);
      break;
    case loco::DataType::S16:
      symmetric_wquant_with_minmax_per_layer(node, min, max, scaling_factor, zp, nudged_min,
                                             nudged_max);
      break;
    default:
      throw std::runtime_error("Unsupported data type");
  }

  auto quantparam = std::make_unique<luci::CircleQuantParam>();
  quantparam->scale.push_back(scaling_factor);
  quantparam->zerop.push_back(zp);
  node->quantparam(std::move(quantparam));
}

/**
 * tells if pad_v2 quantization should ignore padding value
 * In that case padding const will be quantized with input parameters, and probably clipped
 */
bool ignore_pad_v2_const_quantization(luci::CirclePadV2 *pad)
{
  // This is a workaround to quantize pad generated from MaxPoolWithArgmax operation properly
  // TODO use metadata hints to detect this case
  auto const_value_node = dynamic_cast<luci::CircleConst *>(pad->arg(2));
  if (!const_value_node)
    return false;
  if (const_value_node->dtype() == loco::DataType::FLOAT32)
  {
    float const_value = const_value_node->at<loco::DataType::FLOAT32>(0);
    if (const_value == std::numeric_limits<float>::lowest())
      return true;
  }
  return false;
}

void overwrite_quantparam(luci::CircleNode *source, luci::CircleNode *target)
{
  auto source_qparam = source->quantparam();
  if (source_qparam == nullptr)
    throw std::runtime_error("source quantparam is not found during overwrite");

  auto target_qparam = target->quantparam();
  if (target_qparam == nullptr)
  {
    auto quantparam = std::make_unique<luci::CircleQuantParam>();
    target->quantparam(std::move(quantparam));
    target_qparam = target->quantparam();

    if (target_qparam == nullptr)
      throw std::runtime_error("Creating new quant param failed");
  }
  target_qparam->min = source_qparam->min;
  target_qparam->max = source_qparam->max;
  target_qparam->scale = source_qparam->scale;
  target_qparam->zerop = source_qparam->zerop;
  target_qparam->quantized_dimension = source_qparam->quantized_dimension;
}

/** EXAMPLE
 *
 * BEFORE
 *
 *         [CircleNode]       [CircleConst]
 *           (qparam1)           (FP32)
 *                   \            /
 *                    \          /
 *                    [CirclePack]
 *                     (qparam2)
 *
 *  AFTER
 *
 *         [CircleNode]        [CircleConst]   [CircleConst] <- Dead node
 *           (qparam2)           (qparam2)         (FP32)
 *                   \            /
 *                    \          /
 *                    [CirclePack]
 *                     (qparam2)
 *
 * NOTE Quantization parameter of CirclePack (qparam2) is propagated to the inputs.
 */
void propagate_pack_quantparam(luci::CirclePack *pack, loco::DataType quant_type)
{
  assert(pack->quantparam() != nullptr);

  const auto num_inputs = pack->values_count();

  for (uint32_t i = 0; i < num_inputs; i++)
  {
    auto node = loco::must_cast<luci::CircleNode *>(pack->arg(i));

    // Skip if this input is PACK Op
    if (node->opcode() == luci::CircleOpcode::PACK)
      continue;

    // Quantize constant values
    if (node->opcode() == luci::CircleOpcode::CIRCLECONST)
    {
      luci::CircleConst *const_node = loco::must_cast<luci::CircleConst *>(node);
      if (const_node->dtype() != loco::DataType::FLOAT32)
        throw std::runtime_error("Unsupported data type for constant input of pack Op");

      const auto pack_qparam = pack->quantparam();
      if (pack_qparam == nullptr)
        throw std::runtime_error("quantparam of pack is not found during propagation");

      assert(pack_qparam->scale.size() == 1);
      assert(pack_qparam->zerop.size() == 1);
      const auto scaling_factor = pack_qparam->scale[0];
      const auto zerop = pack_qparam->zerop[0];

      auto new_const = luci::clone(const_node);
      quant_const_values(new_const, scaling_factor, zerop, quant_type);
      pack->values(i, new_const);
      overwrite_quantparam(pack, new_const);
    }
    else
    {
      const auto succs = loco::succs(node);
      if (succs.size() > 1)
        continue;

      // Non-const input must have been quantized
      assert(node->quantparam() != nullptr);
      overwrite_quantparam(pack, node);
    }
  }
}

/** EXAMPLE
 *
 *
 *
 * BEFORE
 *
 *      [CircleNode] [CircleConst] [CircleConst] [CircleNode]
 *          (S32)        (S32)        (FP32)     (U8 qparam1)
 *              \          \           /            /
 *               \          \        /            /
 *                \          \     /            /
 *                 -------[CircleOneHot]-------
 *                         (U8 qparam2)
 *
 *  AFTER
 *
 *      [CircleNode] [CircleConst] [CircleConst] [CircleNode]      [CircleConst] <- Dead node
 *          (S32)        (S32)     (U8 qparam2)  (U8 qparam2)         (FP32)
 *              \          \           /           /
 *               \          \        /            /
 *                \          \     /            /
 *                 -------[CircleOneHot]-------
 *                         (U8 qparam2)
 *
 * NOTE Quantization parameter of CircleOneHot (qparam2) is propagated to on_value/off_value.
 */
void propagate_one_hot_quantparam(luci::CircleOneHot *one_hot, loco::DataType quant_type)
{
  assert(one_hot->quantparam() != nullptr);

  // Propagate quantization parameters from output to inputs,
  // to fit both input and counstant_value in one quant range.
  auto quant_input = [one_hot, quant_type](void (luci::CircleOneHot::*arg_setter)(loco::Node *),
                                           loco::Node *(luci::CircleOneHot::*arg_getter)() const) {
    auto node = loco::must_cast<luci::CircleNode *>((one_hot->*arg_getter)());

    // Quantize constant values
    if (node->opcode() == luci::CircleOpcode::CIRCLECONST)
    {
      luci::CircleConst *const_node = loco::must_cast<luci::CircleConst *>(node);
      if (is_quantized(const_node))
        return;

      if (const_node->dtype() != loco::DataType::FLOAT32)
        throw std::runtime_error("Unsupported data type for constant input of OneHot Op");

      const auto qparam = one_hot->quantparam();
      if (qparam == nullptr)
        throw std::runtime_error("quantparam of OneHot is not found during propagation");

      assert(qparam->scale.size() == 1);
      const auto scaling_factor = qparam->scale.at(0);
      const auto zerop = qparam->zerop.at(0);

      auto new_const = luci::clone(const_node);
      quant_const_values(new_const, scaling_factor, zerop, quant_type);
      overwrite_quantparam(one_hot, new_const);
      (one_hot->*arg_setter)(new_const);
    }
    // Subsequent OneHot Ops quant params are not propagated
    else if (node->opcode() == luci::CircleOpcode::ONE_HOT)
    {
      return;
    }
    else
    {
      const auto succs = loco::succs(node);
      if (succs.size() > 1)
        return;

      // Non-const input must have been quantized
      assert(node->quantparam() != nullptr);
      overwrite_quantparam(one_hot, node);
    }
  };

  quant_input(&luci::CircleOneHot::on_value, &luci::CircleOneHot::on_value);
  quant_input(&luci::CircleOneHot::off_value, &luci::CircleOneHot::off_value);
}

} // namespace

namespace luci
{

/** BEFORE
 *
 *         [CircleNode]             [CircleConst]
 *         (U8 qparam1)                 (FP32)
 *                   \                    /
 *                    \                  /
 *                    [CircleConcatenation]
 *                        (U8 qparam2)
 *
 *  AFTER
 *         [CircleNode]             [CircleConst]   [CircleConst] <- Dead node
 *         (U8 qparam2)             (U8 qparam2)       (FP32)
 *                   \                    /
 *                    \                  /
 *                    [CircleConcatenation]
 *                        (U8 qparam2)
 */
void propagate_concat_quantparam(luci::CircleConcatenation *concat, loco::DataType quant_type)
{
  assert(concat->quantparam() != nullptr);

  const auto num_inputs = concat->numValues();

  // Quantize const inputs using their values if concat has fused act function
  if (concat->fusedActivationFunction() != luci::FusedActFunc::NONE)
  {
    for (uint32_t i = 0; i < num_inputs; i++)
    {
      auto node = concat->arg(i);
      auto const_node = dynamic_cast<luci::CircleConst *>(node);
      if (const_node != nullptr)
      {
        auto new_const = luci::clone(const_node);
        quant_const(new_const, quant_type);
        concat->values(i, new_const);
      }
    }
    return;
  }

  for (uint32_t i = 0; i < num_inputs; i++)
  {
    auto node = loco::must_cast<luci::CircleNode *>(concat->arg(i));

    // Skip if this input is CONCAT Op
    if (node->opcode() == luci::CircleOpcode::CONCATENATION)
      continue;

    // Quantize constant values
    if (node->opcode() == luci::CircleOpcode::CIRCLECONST)
    {
      luci::CircleConst *const_node = loco::must_cast<luci::CircleConst *>(node);

      const auto concat_qparam = concat->quantparam();
      assert(concat_qparam->scale.size() == 1);
      const auto scaling_factor = concat_qparam->scale[0];
      const auto zerop = concat_qparam->zerop[0];

      auto new_const = luci::clone(const_node);
      quant_const_values(new_const, scaling_factor, zerop, quant_type);
      concat->values(i, new_const);
      overwrite_quantparam(concat, new_const);
    }
    else
    {
      const auto succs = loco::succs(node);
      if (succs.size() > 1)
        continue;

      // Non-const input must have been quantized
      assert(node->quantparam() != nullptr);
      overwrite_quantparam(concat, node);
    }
  }
}

/** BEFORE
 *
 *         [CircleNode] [CircleConst] [CircleConst]
 *         (U8 qparam1)     (S32)       (FP32)
 *                   \        |         /
 *                    \       |        /
 *                      [CirclePadV2]
 *                       (U8 qparam2)
 *
 *  AFTER (case 1)
 *
 *  By default qparam is propagated from output to inputs to meet backend requirements.
 *
 *         [CircleNode] [CircleConst] [CircleConst]   [CircleConst] <- Dead node
 *         (U8 qparam2)     (S32)      (U8 qparam2)       (FP32)
 *                   \        |         /
 *                    \       |        /
 *                      [CirclePadV2]
 *                       (U8 qparam2)
 *
 *  AFTER (case 2)
 *
 * In case padded value is the lowest float value
 * Qparam is propagated from input to output and constant.
 *
 * This is a special case for optimization constructed pad, needed to guarantee that
 * extremely large negative constant do not stretch output quantization range.
 *
 *         [CircleNode] [CircleConst] [CircleConst]   [CircleConst] <- Dead node
 *         (U8 qparam1)     (S32)      (U8 qparam1)       (FP32)
 *                   \        |         /
 *                    \       |        /
 *                      [CirclePadV2]
 *                       (U8 qparam1)
 */
void propagate_pad_v2_quantparam(luci::CirclePadV2 *pad_v2, loco::DataType quant_type)
{
  if (ignore_pad_v2_const_quantization(pad_v2))
  {
    // propagate input quantization paramters from input to output and padding const value
    auto pad_v2_input = loco::must_cast<luci::CircleNode *>(pad_v2->arg(0));
    overwrite_quantparam(pad_v2_input, pad_v2);

    auto const_value_node = loco::must_cast<luci::CircleConst *>(
      pad_v2->arg(2)); // FIX ignore_pad_v2_const_quantization UNLESS
    auto new_const = luci::clone(const_value_node);

    const auto pad_v2_input_qparam = pad_v2_input->quantparam();
    assert(pad_v2_input_qparam != nullptr);
    assert(pad_v2_input_qparam->scale.size() == 1);
    const auto scaling_factor = pad_v2_input_qparam->scale.at(0);
    const auto zerop = pad_v2_input_qparam->zerop.at(0);

    quant_const_values(new_const, scaling_factor, zerop, quant_type);
    overwrite_quantparam(pad_v2_input, new_const);
    pad_v2->constant_values(new_const);
    return;
  }

  // Propagate quantization paramters from output to inputs,
  // to fit both input and counstant_value in one quant range.
  auto quant_input = [pad_v2, quant_type](void (luci::CirclePadV2::*arg_setter)(loco::Node *),
                                          uint32_t arg) {
    auto node = loco::must_cast<luci::CircleNode *>(pad_v2->arg(arg));

    // Quantize constant values
    if (node->opcode() == luci::CircleOpcode::CIRCLECONST)
    {
      luci::CircleConst *const_node = loco::must_cast<luci::CircleConst *>(node);
      if (is_quantized(const_node))
        return;

      if (const_node->dtype() != loco::DataType::FLOAT32)
        throw std::runtime_error("Unsupported data type for constant input of PadV2 Op");

      const auto pad_v2_qparam = pad_v2->quantparam();
      if (pad_v2_qparam == nullptr)
        throw std::runtime_error("quantparam of PadV2 is not found during propagation");

      assert(pad_v2_qparam->scale.size() == 1);
      const auto scaling_factor = pad_v2_qparam->scale.at(0);
      const auto zerop = pad_v2_qparam->zerop.at(0);

      auto new_const = luci::clone(const_node);
      quant_const_values(new_const, scaling_factor, zerop, quant_type);
      overwrite_quantparam(pad_v2, new_const);
      (pad_v2->*arg_setter)(new_const);
    }
    // Subsequent PadV2 Ops quant params are not propagated
    else if (node->opcode() == luci::CircleOpcode::PADV2)
    {
      return;
    }
    else
    {
      const auto succs = loco::succs(node);
      if (succs.size() > 1)
        return;

      // Non-const input must have been quantized
      assert(node->quantparam() != nullptr);
      overwrite_quantparam(pad_v2, node);
    }
  };

  quant_input(&luci::CirclePadV2::input, 0);
  quant_input(&luci::CirclePadV2::constant_values, 2);
}

} // namespace luci

namespace
{

// Visitor to propagate quantization parameters backwards
struct PropagateQParamBackward final : public luci::CircleNodeMutableVisitor<void>
{
  PropagateQParamBackward(loco::DataType output) : _output_type(output) {}

private:
  loco::DataType _output_type;

  void visit(luci::CircleNode *) {}

  void visit(luci::CircleConcatenation *node) { propagate_concat_quantparam(node, _output_type); }

  void visit(luci::CirclePadV2 *node) { propagate_pad_v2_quantparam(node, _output_type); }

  void visit(luci::CirclePack *node) { propagate_pack_quantparam(node, _output_type); }

  void visit(luci::CircleOneHot *node) { propagate_one_hot_quantparam(node, _output_type); }
};

} // namespace

namespace luci
{

bool PropagateQParamBackwardPass::run(loco::Graph *g)
{
  LOGGER(l);

  // We use post-order traversal as qparam is propagated backward
  for (auto node : loco::postorder_traversal(loco::output_nodes(g)))
  {
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    INFO(l) << "PropagateQParamBackwardPass visit node: " << circle_node->name() << std::endl;

    PropagateQParamBackward pqb(_output_model_dtype);
    circle_node->accept(&pqb);
  }

  return false;
}

} // namespace luci
