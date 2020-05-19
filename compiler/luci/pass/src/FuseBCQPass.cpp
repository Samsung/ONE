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

namespace
{

const std::string node_name_prefix(luci::NodeName node_name, bool origin)
{
  std::string prefix = node_name;

  if (origin)
  {
    const auto first_slash = prefix.find_last_of("/");
    prefix = prefix.substr(0, first_slash);
  }

  const auto second_slash = prefix.find_last_of("/");
  prefix = prefix.substr(0, second_slash + 1);

  const auto start_index = prefix.find("parallel_");
  if (start_index != std::string::npos)
  {
    const auto end_index = prefix.find("body", start_index) + 5;
    const auto left_prefix = prefix.substr(0, start_index);
    const auto right_prefix = prefix.substr(end_index);

    prefix = left_prefix + right_prefix;
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
    const auto prefix = node_name_prefix(node_name, false);

    // Be careful, '_alphas_const' and '_bits_const' appear twice
    if (node_name.find("_num_quantized_bits_const") != std::string::npos)
      const_nodes_num_quantized_bits_const[prefix] = node;
    else if (node_name.find("_per_col_alphas_const") != std::string::npos)
      const_nodes_per_col_alphas_const[prefix] = node;
    else if (node_name.find("_alphas_const") != std::string::npos)
      const_nodes_alphas_const[prefix] = node;
    else if (node_name.find("_bits_const") != std::string::npos)
      const_nodes_bits_const[prefix] = node;
  }

  bool has_BCQ_info_nodes(luci::CircleConst *node)
  {
    const auto prefix = node_name_prefix(node->name(), true);
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
    const auto prefix = node_name_prefix(node->name(), true);
    return const_nodes_alphas_const[prefix];
  }

  luci::CircleConst *get_binary(luci::CircleConst *node)
  {
    const auto prefix = node_name_prefix(node->name(), true);
    return const_nodes_bits_const[prefix];
  }

  luci::CircleConst *get_packed_bits(luci::CircleConst *node)
  {
    const auto prefix = node_name_prefix(node->name(), true);
    auto bits_const = const_nodes_bits_const[prefix];

    const auto graph = node->graph();
    auto packed_bits = graph->nodes()->create<luci::CircleConst>();

    auto dim_bits = bits_const->dim(0).value();
    auto dim_out = bits_const->dim(1).value();
    auto dim_hidden = bits_const->dim(2).value();

    if (dim_hidden % 32 != 0)
      throw std::runtime_error("hidden size must be multiplier of 32");

    // [bits][out][hidden] --> [out][bits][hidden/32]
    packed_bits->dtype(loco::DataType::S32);
    packed_bits->rank(3);
    packed_bits->dim(0) = dim_out;
    packed_bits->dim(1) = dim_bits;
    packed_bits->dim(2) = dim_hidden / 32;

    packed_bits->size<loco::DataType::S32>(dim_out * dim_bits * (dim_hidden / 32));

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
          packed_bits->at<loco::DataType::S32>(o * dim_bits * (dim_hidden / 32) +
                                               b * (dim_hidden / 32) + h) = bits;
        }
      }
    }

    return packed_bits;
  }

  void clear_BCQ_nodes()
  {
    auto createNoOp = [](luci::CircleConst *const_node) {
      auto graph = const_node->graph();
      auto noOp = graph->nodes()->create<luci::CircleOutputExclude>();

      noOp->dtype(const_node->dtype());

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
      if (params != nullptr && converter.has_BCQ_info_nodes(params))
      {
        auto bcq_gather = g->nodes()->create<luci::CircleBCQGather>();

        bcq_gather->input_scales(converter.get_alphas(params));
        bcq_gather->input_binary(converter.get_packed_bits(params));
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
      if (weights != nullptr && converter.has_BCQ_info_nodes(weights))
      {
        //--------------------------------------------------------//
        // Validate quantized weights and information
        auto alphas = converter.get_alphas(weights);
        auto bins = converter.get_binary(weights);
        int out_size = alphas->dim(0).value();
        int hidden_size = converter.get_packed_bits(weights)->dim(2).value() * 32;

        for (int i = 0; i < out_size; ++i)
          for (int j = 0; j < hidden_size; ++j)
          {
            float res = 0.0;
            for (int a = 0; a < 3; ++a)
            {
              if (bins->at<loco::DataType::BOOL>(a * out_size * hidden_size + i * hidden_size +
                                                 j) == 1)
                res += alphas->at<loco::DataType::FLOAT32>(i * 3 + a);
              else
                res -= alphas->at<loco::DataType::FLOAT32>(i * 3 + a);
            }
            // printf("[%d][%d] %f
            // %f\n",i,j,res,weights->at<loco::DataType::FLOAT32>(i*hidden_size+j));
            auto diff = res - weights->at<loco::DataType::FLOAT32>(i * hidden_size + j);
            if (diff < 0)
              diff = -diff;
            assert(diff <= 0.0001);
          }

        auto pack_bins = converter.get_packed_bits(weights);
        for (int i = 0; i < out_size; ++i)
          for (int x = 0; x < 3; ++x)
            for (int j = 0; j < hidden_size / 32; ++j)
            {
              for (int b = 0; b < 32; ++b)
              {
                auto lhs = pack_bins->at<loco::DataType::S32>(i * 3 * hidden_size / 32 +
                                                              x * hidden_size / 32 + j);
                auto lhs_v = (lhs >> b) & 0x1;
                auto rhs = bins->at<loco::DataType::BOOL>(x * out_size * hidden_size +
                                                          i * hidden_size + j * 32 + b);
                // printf("out bits hidden [%d][%d][%d] %d %d %d\n",i,x,j*32+b,lhs,lhs_v,rhs);
                assert(lhs_v == rhs);
              }
            }
        //--------------------------------------------------------//

        auto bcq_fc = g->nodes()->create<luci::CircleBCQFullyConnected>();

        bcq_fc->input(fully_connected->input());
        bcq_fc->weights_scales(converter.get_alphas(weights));
        bcq_fc->weights_binary(converter.get_packed_bits(weights));
        if (dynamic_cast<luci::CircleOutputExclude *>(fully_connected->bias()))
        {
          auto const_node = g->nodes()->create<luci::CircleConst>();
          const_node->dtype(loco::DataType::FLOAT32);
          const_node->rank(1);
          const_node->dim(0) = 1;
          const_node->size<loco::DataType::FLOAT32>(1);
          const_node->at<loco::DataType::FLOAT32>(0) = 0;

          bcq_fc->bias(const_node);
        }
        else
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
