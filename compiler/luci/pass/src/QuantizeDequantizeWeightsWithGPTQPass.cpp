/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved.
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

#include "luci/Pass/QuantizeDequantizeWeightsWithGPTQPass.h"
#include "helpers/LayerInfoMap.h"

#include <luci/IR/CircleNodeVisitor.h>
#include <luci/Service/Nodes/CircleConst.h>
#include <luci/Log.h>

namespace luci
{

namespace
{

/**
 * @brief QuantizeWeightsWithGPTQ quantizes and dequantizes tensors for weights uisng GPTQ algorithm
 * @details Compensate for the quantization error and update weights using Hessian matrix
 *
 */
class QuantizeDequantizeWeightsWithGPTQ final : public luci::CircleNodeMutableVisitor<void>
{
public:
  QuantizeDequantizeWeightsWithGPTQ(
    loco::DataType input, loco::DataType output, QuantizationGranularity granularity,
    std::unordered_map<const luci::CircleNode *, std::vector<float>> *hessian_map)
    : _input_type(input), _output_type(output), _granularity(granularity), _hessian_map(hessian_map)
  {
  }

private:
  loco::DataType _input_type;
  loco::DataType _output_type;
  QuantizationGranularity _granularity;
  std::unordered_map<const luci::CircleNode *, std::vector<float>> *_hessian_map;

  void fake_quantize(luci::CircleConst *weights)
  {
    // To be implemented
    (void)weights;
  }
  void fake_quantize_with_gptq(luci::CircleConst *weights, std::vector<float> &hessian)
  {
    // To be implemented
    (void)weights;
    (void)hessian;
  }

private:
  // Check if
  // 1. node is const
  // 2. node's dtype is float32
  bool is_quantizable(loco::Node *node)
  {
    auto const_node = dynamic_cast<luci::CircleConst *>(node);
    if (not const_node)
      return false;

    // Skip if this is not float32
    if (const_node->dtype() != loco::DataType::FLOAT32)
      return false;

    return true;
  }

  // Default behavior (Do nothing)
  void visit(luci::CircleNode *) {}

  void visit(luci::CircleConv2D *node)
  {
    LOGGER(l);
    INFO(l) << "QuantizeDequantizeWeightsWithGPTQPass visit node: " << node->name() << std::endl;

    if (not is_quantizable(node->filter()))
      return;

    auto weights = loco::must_cast<luci::CircleConst *>(node->filter());
    auto new_weights = luci::clone(weights);
    node->filter(new_weights);

    auto hessian = (*_hessian_map)[node];

    fake_quantize_with_gptq(new_weights, hessian);
  }

  void visit(luci::CircleDepthwiseConv2D *node)
  {
    LOGGER(l);
    INFO(l) << "QuantizeDequantizeWeightsWithGPTQPass visit node: " << node->name() << std::endl;

    if (not is_quantizable(node->filter()))
      return;

    auto weights = loco::must_cast<luci::CircleConst *>(node->filter());
    auto new_weights = luci::clone(weights);
    node->filter(new_weights);

    fake_quantize(new_weights);
  }

  void visit(luci::CircleTransposeConv *node)
  {
    LOGGER(l);
    INFO(l) << "QuantizeDequantizeWeightsWithGPTQPass visit node: " << node->name() << std::endl;

    if (not is_quantizable(node->filter()))
      return;

    auto weights = loco::must_cast<luci::CircleConst *>(node->filter());
    auto new_weights = luci::clone(weights);
    node->filter(new_weights);

    fake_quantize(new_weights);
  }

  void visit(luci::CircleFullyConnected *node)
  {
    LOGGER(l);
    INFO(l) << "QuantizeDequantizeWeightsWithGPTQPass visit node: " << node->name() << std::endl;
    if (not is_quantizable(node->weights()))
      return;

    auto weights = loco::must_cast<luci::CircleConst *>(node->weights());
    auto new_weights = luci::clone(weights);
    node->weights(new_weights);

    auto hessian = (*_hessian_map)[node];

    fake_quantize_with_gptq(new_weights, hessian);
  }
};

} // namespace

bool QuantizeDequantizeWeightsWithGPTQPass::run(loco::Graph *g)
{
  LOGGER(l);
  INFO(l) << "QuantizeDequantizeWeightsWithGPTQ Start" << std::endl;

  if (_ctx->input_model_dtype != loco::DataType::FLOAT32)
    throw std::runtime_error("GPTQPass: Weights-only quantization supports float32 input only");

  if (_ctx->output_model_dtype != loco::DataType::U8 &&
      _ctx->output_model_dtype != loco::DataType::U4)
  {
    throw std::runtime_error("GPTQPass: GPTQ quantization supports uint4/uint8");
  }

  auto info_by_name = layer_info_map(g, _ctx->layers_info);

  auto quantize_dtype = [&](const luci::CircleNode *node) {
    auto iter = info_by_name.find(node->name());

    // Return designated quantization dtype
    if (iter != info_by_name.end())
      return iter->second.dtype;

    // Return default quantization dtype
    return _ctx->output_model_dtype;
  };

  auto quantize_granularity = [&](const luci::CircleNode *node) {
    auto iter = info_by_name.find(node->name());

    // Return designated quantization granularity
    if (iter != info_by_name.end())
      return iter->second.granularity;

    // Return default quantization granularity
    return _ctx->granularity;
  };

  // Quantize weights
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    QuantizeDequantizeWeightsWithGPTQ qw(_ctx->input_model_dtype, quantize_dtype(circle_node),
                                         quantize_granularity(circle_node), _hessian_map);
    circle_node->accept(&qw);
  }

  INFO(l) << "QuantizeDequantizeWeightsWithGPTQ End" << std::endl;
  return false; // one time run
}

} // namespace luci
