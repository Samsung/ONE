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
#include <luci/Log.h>

#include <cassert>
#include <set>

namespace
{

// V means the version of BCQ.
template <int32_t V> class BCQFuser;

template <> class BCQFuser<1>
{
public:
  BCQFuser<1>(int32_t original_output_cnt, int32_t bundle_cnt)
      : _original_output_cnt{original_output_cnt}, _bundle_cnt{bundle_cnt}
  {
    // Do nothing
  }

public:
  void register_bcq_info(loco::Graph *g)
  {
    for (auto node : loco::output_nodes(g))
    {
      auto output_node = loco::must_cast<luci::CircleOutput *>(node);

      /**
       * First output of model is metadata for BCQ. Please refer to following example.
       *
       * When original_output_cnt is 2,
       * BCQ_METADATA, original_output_1, original_output_2, BCQ_INFO_1, ...
       */
      if ((int)output_node->index() > _original_output_cnt)
      {
        const auto prefix = (output_node->index() - (_original_output_cnt + 1)) / (_bundle_cnt);
        const MetadataType metadata_type = static_cast<MetadataType>(
            (output_node->index() - (_original_output_cnt + 1)) % (_bundle_cnt));
        const auto circle_node = loco::must_cast<luci::CircleNode *>(output_node->from());
        add_BCQ_info_node(prefix, metadata_type, circle_node);
      }
    }
  }

  bool fuseBCQ(loco::Graph *g)
  {
    if (!is_bcqinfo_valid())
      return false;

    for (auto f : _fusable_op)
    {
      auto prefix = f.first;
      luci::CircleNode *node = f.second;

      if (!is_valid_prefix(prefix))
        continue;

      // Fuse Gather to BCQGather
      if (auto gather = dynamic_cast<luci::CircleGather *>(node))
      {
        if (auto params = dynamic_cast<luci::CircleConst *>(gather->params()))
        {
          auto bcq_gather = g->nodes()->create<luci::CircleBCQGather>();

          bcq_gather->op_version(1);
          bcq_gather->input_scales(alpha(g, prefix));
          bcq_gather->input_binary(packed_binary_code(g, prefix));
          bcq_gather->indices(gather->indices());
          bcq_gather->input_clusters(packed_clusters(g, prefix));

          if (_do_w_x[prefix]->at<loco::DataType::BOOL>(0))
          {
            bcq_gather->input_hidden_size(params->dim(1).value());
            bcq_gather->axis(gather->axis());
            loco::replace(gather).with(bcq_gather);
          }
          else
          {
            bcq_gather->input_hidden_size(params->dim(0).value());
            const auto axis_transpose = (gather->axis() == 0) ? 1 : 0;
            bcq_gather->axis(axis_transpose);

            const auto indices_rank =
                loco::must_cast<luci::CircleNode *>(gather->indices())->rank();

            auto perm = g->nodes()->create<luci::CircleConst>();
            perm->dtype(loco::DataType::S32);
            perm->size<loco::DataType::S32>(1 + indices_rank);
            perm->rank(1);
            perm->dim(0) = 1 + indices_rank;
            for (uint32_t idx = 0; idx < indices_rank; ++idx)
              perm->at<loco::DataType::S32>(idx) = idx + 1;
            perm->at<loco::DataType::S32>(indices_rank) = 0;
            perm->shape_status(luci::ShapeStatus::VALID);

            auto output_transpose = g->nodes()->create<luci::CircleTranspose>();
            output_transpose->a(bcq_gather);
            output_transpose->perm(perm);

            loco::replace(gather).with(output_transpose);
          }

          return true;
        }
      }

      // Einsum is unpacked to FullyConnected, Pack and Reshape
      if (auto reshape = dynamic_cast<luci::CircleReshape *>(node))
      {
        node = dynamic_cast<luci::CircleNode *>(reshape->tensor());
      }
      if (auto pack = dynamic_cast<luci::CirclePack *>(node))
      {
        if (pack->values_count() == 1 && pack->rank() == 3)
        {
          node = dynamic_cast<luci::CircleNode *>(pack->values(0));
        }
      }

      // Fuse FullyConnected to BCQFullyConnected
      if (auto fully_connected = dynamic_cast<luci::CircleFullyConnected *>(node))
      {
        if (auto weights = dynamic_cast<luci::CircleConst *>(fully_connected->weights()))
        {
          auto bcq_fc = g->nodes()->create<luci::CircleBCQFullyConnected>();

          bcq_fc->op_version(1);
          bcq_fc->weights_scales(alpha(g, prefix));
          bcq_fc->weights_binary(packed_binary_code(g, prefix));
          bcq_fc->bias(fully_connected->bias());
          bcq_fc->weights_clusters(packed_clusters(g, prefix));
          bcq_fc->fusedActivationFunction(fully_connected->fusedActivationFunction());

          loco::Node *bcq_input = fully_connected->input();

          // If input of BCQFullyConnected has more than rank 2, we should reshape it as rank 2
          const auto original_input = loco::must_cast<luci::CircleNode *>(fully_connected->input());
          if (original_input->shape_status() == luci::ShapeStatus::VALID &&
              original_input->rank() > 2)
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
            new_shape->shape_status(luci::ShapeStatus::VALID);

            auto reshape = g->nodes()->create<luci::CircleReshape>();
            reshape->tensor(original_input);
            reshape->shape(new_shape);

            bcq_input = reshape;
          }

          // If x_w formation, we should insert Transpose in front and back of BCQFullyConnected
          if (_do_w_x[prefix]->at<loco::DataType::BOOL>(0))
          {
            bcq_fc->weights_hidden_size(weights->dim(0).value());
            bcq_fc->input(bcq_input);
            loco::replace(fully_connected).with(bcq_fc);
          }
          else
          {
            bcq_fc->weights_hidden_size(weights->dim(1).value());

            auto perm = g->nodes()->create<luci::CircleConst>();
            perm->dtype(loco::DataType::S32);
            perm->size<loco::DataType::S32>(2);
            perm->rank(1);
            perm->dim(0) = 2;
            perm->at<loco::DataType::S32>(0) = 1;
            perm->at<loco::DataType::S32>(1) = 0;
            perm->shape_status(luci::ShapeStatus::VALID);

            auto input_transpose = g->nodes()->create<luci::CircleTranspose>();
            input_transpose->a(bcq_input);
            input_transpose->perm(perm);

            bcq_fc->input(input_transpose);

            auto output_transpose = g->nodes()->create<luci::CircleTranspose>();
            output_transpose->a(bcq_fc);
            output_transpose->perm(perm);

            loco::replace(fully_connected).with(output_transpose);
          }

          return true;
        }
        else
        {
          // TODO Is there any case that input() is constant, instead of weights()?
        }
      }
    }

    return false;
  }

private:
  enum MetadataType
  {
    DO_W_X,
    ALPHA,
    BINARY_CODE,
    NUM_OF_CLUSTERS,
    SIZE_OF_CLUSTERS,
    QBITS_OF_CLUSTERS,
    FUSABLE_OP,
    DEQUANT_WEIGHT,
  };

  void add_BCQ_info_node(int32_t prefix, MetadataType metadata_type, luci::CircleNode *node)
  {
    if (metadata_type == MetadataType::FUSABLE_OP)
    {
      _fusable_op[prefix] = node;
      return;
    }

    luci::CircleConst *const_node;

    // Converter in TensorFlow v1.x sometimes generate Reshape op
    if (auto reshape = dynamic_cast<luci::CircleReshape *>(node))
      const_node = loco::must_cast<luci::CircleConst *>(reshape->tensor());
    else
      const_node = loco::must_cast<luci::CircleConst *>(node);

    if (metadata_type == MetadataType::DO_W_X)
      _do_w_x[prefix] = const_node;
    else if (metadata_type == MetadataType::ALPHA)
      _alpha[prefix] = const_node;
    else if (metadata_type == MetadataType::BINARY_CODE)
      _packed_binary_code[prefix] = const_node;
    else if (metadata_type == MetadataType::NUM_OF_CLUSTERS)
      _number_of_clusters[prefix] = const_node;
    else if (metadata_type == MetadataType::SIZE_OF_CLUSTERS)
      _size_of_clusters[prefix] = const_node;
    else if (metadata_type == MetadataType::QBITS_OF_CLUSTERS)
      _qbits_of_clusters[prefix] = const_node;
    else
      _dequant_weight[prefix] = const_node;
  }

  bool is_bcqinfo_valid()
  {
    LOGGER(l);

    for (auto n : _do_w_x)
    {
      // do_w_x should be BOOL type
      if (n.second->dtype() != loco::DataType::BOOL)
      {
        WARN(l) << "FuseBCQPass : do_w_x has wrong type" << std::endl;
        return false;
      }
    }

    for (auto n : _alpha)
    {
      // alpha should be FLOAT32 type
      if (n.second->dtype() != loco::DataType::FLOAT32)
      {
        WARN(l) << "FuseBCQPass : alpha has wrong type" << std::endl;
        return false;
      }
    }

    for (auto n : _packed_binary_code)
    {
      // packed_binary_code should be INT32 type
      if (n.second->dtype() != loco::DataType::S32)
      {
        WARN(l) << "FuseBCQPass : packed_binary_code has wrong type" << std::endl;
        return false;
      }
    }

    for (auto n : _number_of_clusters)
    {
      // number_of_clusters should be INT32 type
      if (n.second->dtype() != loco::DataType::S32)
      {
        WARN(l) << "FuseBCQPass : number_of_clusters has wrong type" << std::endl;
        return false;
      }
    }

    for (auto n : _size_of_clusters)
    {
      // size_of_clusters should be INT32 type
      if (n.second->dtype() != loco::DataType::S32)
      {
        WARN(l) << "FuseBCQPass : size_of_clusters has wrong type" << std::endl;
        return false;
      }
    }

    for (auto n : _qbits_of_clusters)
    {
      // qbits_of_clusters should be INT32 type
      if (n.second->dtype() != loco::DataType::S32)
      {
        WARN(l) << "FuseBCQPass : qbits_of_clusters has wrong type" << std::endl;
        return false;
      }
    }

    // As dequant_weight is not used for fusing, skip validation.

    return true;
  }

  bool is_valid_prefix(int32_t prefix)
  {
    LOGGER(l);

    if (_do_w_x.find(prefix) == _do_w_x.end())
    {
      WARN(l) << "do_w_x is not found" << std::endl;
      return false;
    }

    if (_alpha.find(prefix) == _alpha.end())
    {
      WARN(l) << "alpha is not found" << std::endl;
      return false;
    }

    if (_packed_binary_code.find(prefix) == _packed_binary_code.end())
    {
      WARN(l) << "packed_binary_code is not found" << std::endl;
      return false;
    }

    if (_number_of_clusters.find(prefix) == _number_of_clusters.end())
    {
      WARN(l) << "number_of_clusters is not found" << std::endl;
      return false;
    }

    if (_size_of_clusters.find(prefix) == _size_of_clusters.end())
    {
      WARN(l) << "size_of_clusters is not found" << std::endl;
      return false;
    }

    if (_qbits_of_clusters.find(prefix) == _qbits_of_clusters.end())
    {
      WARN(l) << "qbits_of_clusters is not found" << std::endl;
      return false;
    }

    // As dequant_weight is not used for fusing, skip validation.

    return true;
  }

private:
  luci::CircleConst *alpha(loco::Graph *graph, int32_t prefix)
  {
    auto new_alpha = graph->nodes()->create<luci::CircleConst>();

    new_alpha->dtype(loco::DataType::FLOAT32);
    new_alpha->size<loco::DataType::FLOAT32>(_alpha[prefix]->size<loco::DataType::FLOAT32>());
    new_alpha->rank(1);
    new_alpha->dim(0) = _alpha[prefix]->dim(0);
    for (uint32_t i = 0; i < _alpha[prefix]->size<loco::DataType::FLOAT32>(); ++i)
      new_alpha->at<loco::DataType::FLOAT32>(i) = _alpha[prefix]->at<loco::DataType::FLOAT32>(i);
    new_alpha->shape_status(luci::ShapeStatus::VALID);

    return new_alpha;
  }

  luci::CircleConst *packed_binary_code(loco::Graph *graph, int32_t prefix)
  {
    auto new_beta = graph->nodes()->create<luci::CircleConst>();

    new_beta->dtype(loco::DataType::S32);
    new_beta->size<loco::DataType::S32>(_packed_binary_code[prefix]->size<loco::DataType::S32>());
    new_beta->rank(2);
    new_beta->dim(0) = _packed_binary_code[prefix]->dim(0);
    new_beta->dim(1) = _packed_binary_code[prefix]->dim(1);
    for (uint32_t i = 0; i < _packed_binary_code[prefix]->size<loco::DataType::S32>(); ++i)
      new_beta->at<loco::DataType::S32>(i) =
          _packed_binary_code[prefix]->at<loco::DataType::S32>(i);
    new_beta->shape_status(luci::ShapeStatus::VALID);

    return new_beta;
  }

  luci::CircleConst *packed_clusters(loco::Graph *graph, int32_t prefix)
  {
    auto qbits_of_clusters = _qbits_of_clusters[prefix];
    auto size_of_clusters = _size_of_clusters[prefix];
    const auto number_of_clusters = _number_of_clusters[prefix]->at<loco::DataType::S32>(0);

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

private:
  std::map<int32_t, luci::CircleConst *> _do_w_x;
  std::map<int32_t, luci::CircleConst *> _alpha;
  std::map<int32_t, luci::CircleConst *> _packed_binary_code;
  std::map<int32_t, luci::CircleConst *> _number_of_clusters;
  std::map<int32_t, luci::CircleConst *> _size_of_clusters;
  std::map<int32_t, luci::CircleConst *> _qbits_of_clusters;
  std::map<int32_t, luci::CircleConst *> _dequant_weight;
  std::map<int32_t, luci::CircleNode *> _fusable_op;

private:
  int32_t _original_output_cnt = 0;
  int32_t _bundle_cnt = 0;
};

} // namespace

namespace luci
{

bool FuseBCQPass::run(loco::Graph *g)
{
  bool changed = false;

  const int32_t start_magicnum = -2e9 + 27;
  const int32_t end_magicnum = 2e9 - 27;

  loco::Graph *main_graph = g;

  luci::CircleConst *metadata_node = nullptr;
  for (auto node : loco::output_nodes(g))
  {
    auto output_node = loco::must_cast<luci::CircleOutput *>(node);

    // Metadata node should be first output
    if (output_node->index() != 0)
      continue;

    // Metadata should be constant and dtype should be S32
    auto const_node = dynamic_cast<luci::CircleConst *>(output_node->from());
    if (const_node == nullptr || const_node->dtype() != loco::DataType::S32)
      continue;

    // Metadata has at least four elements
    const auto element_cnt = const_node->size<loco::DataType::S32>();
    if (element_cnt < 4)
      continue;

    // Metadata has magic numbers at first and at last
    const auto start_value = const_node->at<loco::DataType::S32>(0);
    const auto end_value = const_node->at<loco::DataType::S32>(element_cnt - 1);
    if (start_value == start_magicnum && end_value == end_magicnum)
    {
      metadata_node = const_node;
      break;
    }
  }

  if (metadata_node != nullptr)
  {
    const auto bcq_version = metadata_node->at<loco::DataType::S32>(1);
    const auto original_output_cnt = metadata_node->at<loco::DataType::S32>(2);

    if (bcq_version == 1)
    {
      const auto bundle_cnt = metadata_node->at<loco::DataType::S32>(3);

      BCQFuser<1> fuser{original_output_cnt, bundle_cnt};
      fuser.register_bcq_info(main_graph);

      if (fuser.fuseBCQ(g))
        changed = true;
    }
    else
    {
      LOGGER(l);
      WARN(l) << "Not supported BCQ version is found." << std::endl;
    }

    // Remove all of BCQ information nodes iff there is no change
    if (changed == false)
    {
      for (auto node : loco::output_nodes(main_graph))
      {
        auto output_node = loco::must_cast<luci::CircleOutput *>(node);
        if (output_node->index() == 0 || (int)output_node->index() > original_output_cnt)
        {
          auto noOp = main_graph->nodes()->create<luci::CircleOutputExclude>();
          noOp->dtype(loco::DataType::FLOAT32); // TODO Remove this setting
          output_node->from(noOp);
          changed = true;
        }
      }
    }
  }

  return changed;
}

} // namespace luci
