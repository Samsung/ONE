/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/FuseBCQPass.h"

#include <luci/IR/CircleNodes.h>

#include <cassert>
#include <string>
#include <set>

/**
 * NOTE Current FuseBCQPass is implemented only for special cases.
 *      And schema of BCQ operations is subject to be changed in near future.
 *      Accordingly, most of implementations for FuseBCQPass will be changed.
 *
 * TODO Revise current implementation to support updated BCQ operations.
 */

namespace
{

const std::string node_name_prefix(luci::NodeName node_name)
{
  std::string prefix = node_name;

  if (prefix.find("/while") != std::string::npos)
  {
    const auto index = prefix.find("/while");
    const auto left_prefix = prefix.substr(0, index);
    const auto right_prefix = prefix.substr(index + 6);

    prefix = left_prefix + right_prefix;
  }

  if (prefix.find("parallel_") != std::string::npos)
  {
    const auto start_index = prefix.find("parallel_");

    auto end_index = prefix.find("/", start_index);
    end_index = prefix.find("/", end_index);

    const auto left_prefix = prefix.substr(0, start_index);
    const auto right_prefix = prefix.substr(end_index);

    prefix = left_prefix + right_prefix;
  }

  if (prefix.find("Tensordot/MatMul") != std::string::npos)
  {
    const auto index = prefix.find("Tensordot/MatMul");
    prefix = prefix.substr(0, index - 1);
  }
  else if (prefix.find("GatherV2") != std::string::npos)
  {
    const auto index = prefix.find("GatherV2");
    prefix = prefix.substr(0, index - 1);
  }
  else if (prefix.find("weights/read") != std::string::npos)
  {
    const auto index = prefix.find("weights/read");
    prefix = prefix.substr(0, index - 1);
  }
  else if (prefix.find("kernel_") != std::string::npos)
  {
    const auto index = prefix.find("kernel_");
    prefix = prefix.substr(0, index - 1);
  }
  else if (prefix.find("weights_") != std::string::npos)
  {
    const auto index = prefix.find("weights_");
    prefix = prefix.substr(0, index - 1);
  }

  return prefix;
}

} // namespace

namespace
{

class BCQConverter final
{
public:
  void add_BCQ_info_node(luci::CircleConst *node)
  {
    const auto node_name = node->name();
    const auto prefix = node_name_prefix(node_name);

    // Be careful for order, '_alphas_const' and '_bits_const' appear twice.
    if (node_name.find("_num_quantized_bits_const") != std::string::npos)
      const_nodes_num_quantized_bits_const[prefix] = node;
    else if (node_name.find("_per_col_alphas_const") != std::string::npos)
      const_nodes_per_col_alphas_const[prefix] = node;
    else if (node_name.find("_alphas_const") != std::string::npos)
      const_nodes_alphas_const[prefix] = node;
    else if (node_name.find("_bits_const") != std::string::npos)
      const_nodes_bits_const[prefix] = node;
  }

  bool has_BCQ_info(luci::CircleConst *node)
  {
    const auto prefix = node_name_prefix(node->name());
    bool has_info = true;

    has_info &= (const_nodes_alphas_const.find(prefix) != const_nodes_alphas_const.end());
    has_info &= (const_nodes_bits_const.find(prefix) != const_nodes_bits_const.end());
    has_info &= (const_nodes_num_quantized_bits_const.find(prefix) !=
                 const_nodes_num_quantized_bits_const.end());
    has_info &=
        (const_nodes_per_col_alphas_const.find(prefix) != const_nodes_per_col_alphas_const.end());

    return has_info;
  }

  luci::CircleConst *get_alphas(luci::CircleConst *node)
  {
    const auto prefix = node_name_prefix(node->name());
    return const_nodes_alphas_const[prefix];
  }

  luci::CircleConst *get_binary(luci::CircleConst *node)
  {
    const auto prefix = node_name_prefix(node->name());
    return const_nodes_bits_const[prefix];
  }

  luci::CircleConst *get_packed_binary(luci::CircleConst *node)
  {
    auto bits_const = get_binary(node);

    const auto graph = node->graph();
    auto packed_binary = graph->nodes()->create<luci::CircleConst>();

    auto dim_bits = bits_const->dim(0).value();
    auto dim_out = bits_const->dim(1).value();
    auto dim_hidden = bits_const->dim(2).value();

    if (dim_hidden % 32 != 0)
      throw std::runtime_error("hidden size must be multiplier of 32");

    // TODO Support when hidden size cannot be divided by 32
    packed_binary->dtype(loco::DataType::S32);
    packed_binary->rank(3);
    packed_binary->dim(0) = dim_out;
    packed_binary->dim(1) = dim_bits;
    packed_binary->dim(2) = dim_hidden / 32;

    packed_binary->size<loco::DataType::S32>(dim_out * dim_bits * (dim_hidden / 32));

    for (uint32_t o = 0; o < dim_out; ++o)
    {
      for (uint32_t b = 0; b < dim_bits; ++b)
      {
        for (uint32_t h = 0; h < dim_hidden / 32; ++h)
        {
          int32_t bits = 0;
          for (int32_t i = 31; i >= 0; --i)
          {
            bits = ((bits << 1) | (bits_const->at<loco::DataType::BOOL>(
                                       b * dim_out * dim_hidden + o * dim_hidden + (h * 32 + i)) &
                                   0x1));
          }
          packed_binary->at<loco::DataType::S32>(o * dim_bits * (dim_hidden / 32) +
                                                 b * (dim_hidden / 32) + h) = bits;
        }
      }
    }

    return packed_binary;
  }

  /**
   * @brief Exclude BCQ information nodes which are used for fusing BCQ operations
   *        from graph output by using CircleOutputExclude
   */
  void clear_BCQ_nodes()
  {
    auto createNoOp = [](luci::CircleConst *const_node) {
      auto graph = const_node->graph();
      auto noOp = graph->nodes()->create<luci::CircleOutputExclude>();

      noOp->dtype(const_node->dtype());
      noOp->rank(const_node->rank());
      for (uint32_t i = 0; i < const_node->rank(); ++i)
        noOp->dim(i) = const_node->dim(i);

      return noOp;
    };

    for (auto &n : const_nodes_alphas_const)
    {
      auto node = n.second;

      for (auto s : loco::succs(node))
      {
        if (auto outnode = dynamic_cast<luci::CircleOutput *>(s))
        {
          outnode->from(createNoOp(node));
        }
      }
    }

    for (auto &n : const_nodes_num_quantized_bits_const)
    {
      auto node = n.second;
      loco::replace(node).with(createNoOp(node));
    }

    for (auto &n : const_nodes_bits_const)
    {
      auto node = n.second;
      loco::replace(node).with(createNoOp(node));
    }

    for (auto &n : const_nodes_per_col_alphas_const)
    {
      auto node = n.second;
      loco::replace(node).with(createNoOp(node));
    }
  }

  bool is_valid_BCQ(luci::CircleConst *node)
  {
    auto alphas = get_alphas(node);
    auto binary = get_binary(node);

    if (alphas->dim(0).value() != binary->dim(1).value())
      return false;

    if (alphas->dim(1).value() != binary->dim(0).value())
      return false;

    // TODO Support when hidden size cannot be divided by 32
    if (binary->dim(2).value() % 32 != 0)
      return false;

    int bits_size = alphas->dim(1).value();
    int out_size = alphas->dim(0).value();
    int hidden_size = binary->dim(2).value();

    // Quantized value validation
    for (int o = 0; o < out_size; ++o)
    {
      for (int h = 0; h < hidden_size; ++h)
      {
        float res = 0.0;
        for (int b = 0; b < bits_size; ++b)
        {
          if (binary->at<loco::DataType::BOOL>(b * out_size * hidden_size + o * hidden_size + h) ==
              1)
            res += alphas->at<loco::DataType::FLOAT32>(o * bits_size + b);
          else
            res -= alphas->at<loco::DataType::FLOAT32>(o * bits_size + b);
        }

        auto diff = res - node->at<loco::DataType::FLOAT32>(o * hidden_size + h);
        if (diff < -0.0001 || 0.0001 < diff)
          return false;
      }
    }

    return true;
  }

private:
  std::map<std::string, luci::CircleConst *> const_nodes_alphas_const;
  std::map<std::string, luci::CircleConst *> const_nodes_bits_const;
  std::map<std::string, luci::CircleConst *> const_nodes_num_quantized_bits_const;
  std::map<std::string, luci::CircleConst *> const_nodes_per_col_alphas_const;
};
}

namespace luci
{

bool FuseBCQPass::run(loco::Graph *g)
{
  BCQConverter converter;

  bool changed = false;

  for (auto node : loco::all_nodes(g))
  {
    if (auto circle_const = dynamic_cast<luci::CircleConst *>(node))
    {
      converter.add_BCQ_info_node(circle_const);
    }
  }

  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    if (auto gather = dynamic_cast<luci::CircleGather *>(node))
    {
      auto params = dynamic_cast<luci::CircleConst *>(gather->params());
      if (params != nullptr && converter.has_BCQ_info(params))
      {
        assert(converter.is_valid_BCQ(params));
        auto bcq_gather = g->nodes()->create<luci::CircleBCQGather>();

        bcq_gather->input_scales(converter.get_alphas(params));
        bcq_gather->input_binary(converter.get_packed_binary(params));
        bcq_gather->indices(gather->indices());

        const auto binary_hidden_size =
            dynamic_cast<luci::CircleConst *>(bcq_gather->input_binary())->dim(2).value();
        bcq_gather->input_hidden_size(binary_hidden_size * 32);
        bcq_gather->axis(gather->axis());

        loco::replace(gather).with(bcq_gather);

        changed = true;
      }
    }
    else if (auto fully_connected = dynamic_cast<luci::CircleFullyConnected *>(node))
    {
      auto weights = dynamic_cast<luci::CircleConst *>(fully_connected->weights());
      if (weights != nullptr && converter.has_BCQ_info(weights))
      {
        assert(converter.is_valid_BCQ(weights));

        auto bcq_fc = g->nodes()->create<luci::CircleBCQFullyConnected>();

        bcq_fc->input(fully_connected->input());
        bcq_fc->weights_scales(converter.get_alphas(weights));
        bcq_fc->weights_binary(converter.get_packed_binary(weights));
        bcq_fc->bias(fully_connected->bias());

        const auto binary_hidden_size =
            dynamic_cast<luci::CircleConst *>(bcq_fc->weights_binary())->dim(2).value();
        bcq_fc->weights_hidden_size(binary_hidden_size * 32);
        bcq_fc->fusedActivationFunction(fully_connected->fusedActivationFunction());

        loco::replace(fully_connected).with(bcq_fc);

        changed = true;
      }
    }
  }

  if (changed)
    converter.clear_BCQ_nodes();

  return changed;
}

} // namespace luci
