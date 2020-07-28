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

/**
 * @brief Circle nodes including BCQ information and a circle node to which BCQ will be applied
 *        are connected with their name. And their names include common prefix.
 *        However, after pb file is converted to tflite file, some nodes' name are changed.
 *        Thus this function will return original common prefix.
 *
 * @note  All the re-naming rule of TFLite converter is not figured out.
 *        Therefore, if new naming rule is detected, this function should be updated.
 */
const std::string node_name_prefix(luci::NodeName node_name)
{
  std::string prefix = node_name;

  if (prefix.find("ReadVariableOp/resource/") != std::string::npos)
  {
    const auto start_index = prefix.find("ReadVariableOp/resource/");

    const auto left_prefix = prefix.substr(0, start_index);
    const auto right_prefix = prefix.substr(start_index + 24);

    prefix = left_prefix + right_prefix;
  }

  if (prefix.find("Tensordot/") != std::string::npos)
  {
    const auto index = prefix.find("Tensordot/");
    prefix = prefix.substr(0, index - 1);
  }
  else if (prefix.find("kernel/") != std::string::npos)
  {
    const auto index = prefix.find("kernel/");
    prefix = prefix.substr(0, index - 1);
  }
  else if (prefix.find("/bcqinfo_") != std::string::npos)
  {
    const auto index = prefix.find("/bcqinfo_");
    prefix = prefix.substr(0, index);
  }

  return prefix;
}

/**
 * @brief Create CircleOutputExclude operation, which has same shape and dtype with
 *        original circle_node.
 */
luci::CircleOutputExclude *createNoOp(luci::CircleNode *circle_node)
{
  auto graph = circle_node->graph();
  auto noOp = graph->nodes()->create<luci::CircleOutputExclude>();

  if (circle_node->shape_status() == luci::ShapeStatus::VALID)
  {
    noOp->dtype(circle_node->dtype());
    noOp->rank(circle_node->rank());
    for (uint32_t i = 0; i < circle_node->rank(); ++i)
      noOp->dim(i) = circle_node->dim(i);
  }
  else
  {
    // For type inference
    noOp->dtype(loco::DataType::FLOAT32);
  }

  return noOp;
};

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

    // If bcqinfo_* nodes are held by Reshape operation,
    // shape of bcqinfo_* nodes are copied to `shape` input of Reshape operation.
    // Then the name becomes bcqinfo_*_copy_shape.
    // We should prevent this node not to added to bcq information.
    if (node_name.find("_copy_shape") != std::string::npos)
      return;

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
    else if (node_name.find("bcqinfo_dequant_weight") != std::string::npos)
      _dequant_weight[prefix] = node;
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
    // bcqinfo_dequant_weight is just for validation, so not always exists.

    return has_info;
  }

  bool do_w_x(luci::CircleConst *node)
  {
    const auto prefix = node_name_prefix(node->name());

    if (_do_w_x[prefix]->dtype() == loco::DataType::S32)
      return _do_w_x[prefix]->at<loco::DataType::S32>(0) == 1;
    else
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

  luci::CircleConst *get_number_of_clusters(luci::CircleConst *node)
  {
    const auto prefix = node_name_prefix(node->name());
    return _number_of_clusters[prefix];
  }

  luci::CircleConst *get_size_of_clusters(luci::CircleConst *node)
  {
    const auto prefix = node_name_prefix(node->name());
    return _size_of_clusters[prefix];
  }

  luci::CircleConst *get_qbits_of_clusters(luci::CircleConst *node)
  {
    const auto prefix = node_name_prefix(node->name());
    return _qbits_of_clusters[prefix];
  }

  luci::CircleConst *packed_clusters(luci::CircleConst *node)
  {
    auto graph = node->graph();
    auto qbits_of_clusters = get_qbits_of_clusters(node);
    auto size_of_clusters = get_size_of_clusters(node);
    const auto number_of_clusters = get_number_of_clusters(node)->at<loco::DataType::S32>(0);

    auto packed_clusters = graph->nodes()->create<luci::CircleConst>();
    packed_clusters->dtype(loco::DataType::S32);
    packed_clusters->size<loco::DataType::S32>(number_of_clusters * 2);
    packed_clusters->rank(2);
    packed_clusters->dim(0) = number_of_clusters;
    packed_clusters->dim(1) = 2;
    packed_clusters->shape_status(luci::ShapeStatus::VALID);

    for (int i = 0; i < number_of_clusters; ++i)
    {
      packed_clusters->at<loco::DataType::S32>(i * 2) =
          qbits_of_clusters->at<loco::DataType::S32>(i);
      packed_clusters->at<loco::DataType::S32>(i * 2 + 1) =
          size_of_clusters->at<loco::DataType::S32>(i);
    }

    return packed_clusters;
  }

  /**
   * @brief Exclude BCQ information nodes which are used for fusing BCQ operations
   *        from graph output by using CircleOutputExclude
   */
  void clear_BCQ_nodes()
  {
    auto clear_nodes = [](std::map<std::string, luci::CircleConst *> &nodes) {
      for (auto &n : nodes)
      {
        auto node = n.second;

        for (auto s : loco::succs(node))
        {
          if (auto outnode = dynamic_cast<luci::CircleOutput *>(s))
          {
            outnode->from(createNoOp(node));
          }
          else if (auto reshape_node = dynamic_cast<luci::CircleReshape *>(s))
          {
            for (auto o : loco::succs(reshape_node))
            {
              auto circle_output = loco::must_cast<luci::CircleOutput *>(o);
              circle_output->from(createNoOp(reshape_node));
            }
          }
        }
      }
    };

    clear_nodes(_do_w_x);
    clear_nodes(_alpha);
    clear_nodes(_packed_binary_code);
    clear_nodes(_number_of_clusters);
    clear_nodes(_size_of_clusters);
    clear_nodes(_qbits_of_clusters);
    clear_nodes(_dequant_weight);
  }

  bool is_bcqinfo_valid()
  {
    // do_w_x should be int32 or bool type
    for (auto n : _do_w_x)
    {
      if (n.second->dtype() != loco::DataType::BOOL && n.second->dtype() != loco::DataType::S32)
        return false;
    }

    return true;
  }

private:
  std::map<std::string, luci::CircleConst *> _do_w_x;
  std::map<std::string, luci::CircleConst *> _alpha;
  std::map<std::string, luci::CircleConst *> _packed_binary_code;
  std::map<std::string, luci::CircleConst *> _number_of_clusters;
  std::map<std::string, luci::CircleConst *> _size_of_clusters;
  std::map<std::string, luci::CircleConst *> _qbits_of_clusters;
  std::map<std::string, luci::CircleConst *> _dequant_weight;
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

  if (!converter.is_bcqinfo_valid())
    return false;

  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    if (auto gather = dynamic_cast<luci::CircleGather *>(node))
    {
      auto params = dynamic_cast<luci::CircleConst *>(gather->params());
      if (params != nullptr && converter.has_BCQ_info(params))
      {
        auto bcq_gather = g->nodes()->create<luci::CircleBCQGather>();

        bcq_gather->input_scales(converter.get_alpha(params));
        bcq_gather->input_binary(converter.get_packed_binary_code(params));
        bcq_gather->indices(gather->indices());
        bcq_gather->input_clusters(converter.packed_clusters(params));

        // input_binary shape : [output_size, hidden_size]
        const auto binary_hidden_size =
            loco::must_cast<luci::CircleConst *>(bcq_gather->input_binary())->dim(1).value() * 32;
        bcq_gather->input_hidden_size(binary_hidden_size);

        if (converter.do_w_x(params))
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
        auto bcq_fc = g->nodes()->create<luci::CircleBCQFullyConnected>();

        bcq_fc->weights_scales(converter.get_alpha(weights));
        bcq_fc->weights_binary(converter.get_packed_binary_code(weights));
        bcq_fc->bias(fully_connected->bias());
        bcq_fc->weights_clusters(converter.packed_clusters(weights));
        bcq_fc->fusedActivationFunction(fully_connected->fusedActivationFunction());

        loco::Node *bcq_input = fully_connected->input();
        int32_t batch_rank = 0;

        // If input of BCQFullyConnected has more than rank 2, we should reshape it as rank 2
        const auto original_input = loco::must_cast<luci::CircleNode *>(fully_connected->input());
        if (original_input->shape_status() == ShapeStatus::VALID && original_input->rank() > 2)
        {
          auto new_shape = g->nodes()->create<luci::CircleConst>();
          new_shape->dtype(loco::DataType::S32);
          new_shape->size<loco::DataType::S32>(2);
          new_shape->rank(1);
          new_shape->dim(0) = 2;

          auto batch_size = 1;
          for (uint32_t i = 0; i < original_input->rank() - 1; ++i)
            batch_size *= original_input->dim(i).value();

          new_shape->at<loco::DataType::S32>(0) = batch_size;
          new_shape->at<loco::DataType::S32>(1) =
              original_input->dim(original_input->rank() - 1).value();
          new_shape->shape_status(ShapeStatus::VALID);

          auto reshape = g->nodes()->create<luci::CircleReshape>();
          reshape->tensor(original_input);
          reshape->shape(new_shape);

          bcq_input = reshape;
          batch_rank = original_input->rank() - 2;
        }

        // If x_w formation, we should insert Transpose in front and back of BCQFullyConnected
        if (converter.do_w_x(weights))
        {
          const auto binary_hidden_size =
              loco::must_cast<luci::CircleNode *>(fully_connected->input())
                  ->dim(batch_rank)
                  .value();
          bcq_fc->weights_hidden_size(binary_hidden_size);
          bcq_fc->input(bcq_input);
          loco::replace(fully_connected).with(bcq_fc);
        }
        else
        {
          const auto binary_hidden_size =
              loco::must_cast<luci::CircleNode *>(fully_connected->input())
                  ->dim(1 + batch_rank)
                  .value();
          bcq_fc->weights_hidden_size(binary_hidden_size);

          auto perm = g->nodes()->create<luci::CircleConst>();
          perm->dtype(loco::DataType::S32);
          perm->size<loco::DataType::S32>(2);
          perm->rank(1);
          perm->dim(0) = 2;
          perm->at<loco::DataType::S32>(0) = 1;
          perm->at<loco::DataType::S32>(1) = 0;
          perm->shape_status(ShapeStatus::VALID);

          auto input_transpose = g->nodes()->create<luci::CircleTranspose>();
          input_transpose->a(bcq_input);
          input_transpose->perm(perm);

          bcq_fc->input(input_transpose);

          auto output_transpose = g->nodes()->create<luci::CircleTranspose>();
          output_transpose->a(bcq_fc);
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
