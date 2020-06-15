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

#include "luci/Pass/FuseNewBCQPass.h"

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

/**
 * @brief Circle nodes including BCQ information and a circle node to which BCQ will be applied
 *        are connected with their name. And their names include common prefix.
 *        However, after pb file is converted to tflite file, some nodes' name are changed.
 *        Thus this function will return original common prefix.
 */
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
  else if (prefix.find("MatMul") != std::string::npos)
  {
    const auto index = prefix.find("MatMul");
    prefix = prefix.substr(0, index - 1);
  }
  else if (prefix.find("kernel/") != std::string::npos)
  {
    const auto index = prefix.find("kernel/");
    prefix = prefix.substr(0, index - 1);
  }
  // else if (prefix.find("GatherV2") != std::string::npos)
  // {
  //   const auto index = prefix.find("GatherV2");
  //   prefix = prefix.substr(0, index - 1);
  // }
  // else if (prefix.find("weights/read") != std::string::npos)
  // {
  //   const auto index = prefix.find("weights/read");
  //   prefix = prefix.substr(0, index - 1);
  // }
  // else if (prefix.find("kernel_") != std::string::npos)
  // {
  //   const auto index = prefix.find("kernel_");
  //   prefix = prefix.substr(0, index - 1);
  // }
  // else if (prefix.find("weights_") != std::string::npos)
  // {
  //   const auto index = prefix.find("weights_");
  //   prefix = prefix.substr(0, index - 1);
  // }

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

    if (node_name.find("bcqinfo_do_w_x") != std::string::npos)
      _do_w_x[prefix] = node;
    else if (node_name.find("bcqinfo_alpha") != std::string::npos)
      _alpha[prefix] = node;
    else if (node_name.find("bcqinfo_packed_binary_code") != std::string::npos)
      _packed_binary_code[prefix] = node;
    else if (node_name.find("bcqinfo_number_of_clusters") != std::string::npos)
      _number_of_clusters[prefix] = node;
    else if (node_name.find("bcqinfo_size_of_clusters") != std::string::npos)
      _size_of_clusters[prefix] = node;
    else if (node_name.find("bcqinfo_qbits_of_clusters") != std::string::npos)
      _qbits_of_clusters[prefix] = node;
  }

  bool has_BCQ_info(luci::CircleConst *node)
  {
    const auto prefix = node_name_prefix(node->name());
    bool has_info = true;

    has_info &= (_do_w_x.find(prefix) != _do_w_x.end());
    has_info &= (_alpha.find(prefix) != _alpha.end());
    has_info &= (_packed_binary_code.find(prefix) != _packed_binary_code.end());
    has_info &= (_number_of_clusters.find(prefix) != _number_of_clusters.end());
    has_info &= (_size_of_clusters.find(prefix) != _size_of_clusters.end());
    has_info &= (_qbits_of_clusters.find(prefix) != _qbits_of_clusters.end());

    return has_info;
  }

  bool do_w_x(luci::CircleConst *node)
  {
    const auto prefix = node_name_prefix(node->name());
    return _do_w_x[prefix]->at<loco::DataType::BOOL>(0);
  }

  luci::CircleConst *get_alpha(luci::CircleConst *node)
  {
    const auto prefix = node_name_prefix(node->name());
    return _alpha[prefix];
  }

  luci::CircleConst *get_packed_binary_code(luci::CircleConst *node)
  {
    const auto prefix = node_name_prefix(node->name());
    return _packed_binary_code[prefix];
  }

  luci::CircleConst *get_clusters(luci::CircleConst *node)
  {
    auto graph = node->graph();
    auto clusters = graph->nodes()->create<luci::CircleConst>();
    clusters->rank(2);

    // SHOULD IMPLEMENT THIS!!!!!!!!
    
    return clusters;
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

    for (auto &n : _alpha)
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

    for (auto &n : _packed_binary_code)
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

    for (auto &n : _do_w_x)
    {
      auto node = n.second;
      loco::replace(node).with(createNoOp(node));
    }

    for (auto &n : _number_of_clusters)
    {
      auto node = n.second;
      loco::replace(node).with(createNoOp(node));
    }

    for (auto &n : _size_of_clusters)
    {
      auto node = n.second;
      loco::replace(node).with(createNoOp(node));
    }

    for (auto &n : _qbits_of_clusters)
    {
      auto node = n.second;
      loco::replace(node).with(createNoOp(node));
    }
  }

  bool is_valid_BCQ(luci::CircleConst *)
  {
    return true;
  }

private:
  std::map<std::string, luci::CircleConst *> _do_w_x;
  std::map<std::string, luci::CircleConst *> _alpha;
  std::map<std::string, luci::CircleConst *> _packed_binary_code;
  std::map<std::string, luci::CircleConst *> _number_of_clusters;
  std::map<std::string, luci::CircleConst *> _size_of_clusters;
  std::map<std::string, luci::CircleConst *> _qbits_of_clusters;
};

} // namespace

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

        bcq_gather->input_scales(converter.get_alpha(params));
        bcq_gather->input_binary(converter.get_packed_binary_code(params));
        bcq_gather->indices(gather->indices());
        //bcq_gather->input_clusters

        const auto binary_hidden_size =
            dynamic_cast<luci::CircleConst *>(bcq_gather->input_scales())->dim(1).value();
        bcq_gather->input_hidden_size(binary_hidden_size);

        if(converter.do_w_x(params))
        {
          bcq_gather->axis(gather->axis());
        }
        else
        {
          const auto axis_transpose = (gather->axis() == 0) ? 1 : 0;
          bcq_gather->axis(axis_transpose);
        }
        
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

        bcq_fc->weights_scales(converter.get_alpha(weights));
        bcq_fc->weights_binary(converter.get_packed_binary_code(weights));
        bcq_fc->bias(fully_connected->bias());
        //bcq_fc->weights_clusters

        const auto binary_hidden_size =
            dynamic_cast<luci::CircleConst *>(bcq_fc->weights_scales())->dim(1).value();
        bcq_fc->weights_hidden_size(binary_hidden_size);
        bcq_fc->fusedActivationFunction(fully_connected->fusedActivationFunction());

        if(converter.do_w_x(weights))
        {
          bcq_fc->input(fully_connected->input());
          loco::replace(fully_connected).with(bcq_fc);
        }
        else
        {
          auto perm = g->nodes()->create<luci::CircleConst>();
          perm->rank(2);
          perm->dim(0)=1;
          perm->dim(1)=0;

          auto input_transpose = g->nodes()->create<luci::CircleTranspose>();
          input_transpose->a(fully_connected->input());
          input_transpose->perm(perm);

          bcq_fc->input(input_transpose);

          auto output_transpose = g->nodes()->create<luci::CircleTranspose>();
          output_transpose->a(fully_connected->input());
          output_transpose->perm(perm);

          loco::replace(fully_connected).with(output_transpose);
        }

        changed = true;
      }
    }
  }

  if (changed)
    converter.clear_BCQ_nodes();

  return changed;
}

} // namespace luci
