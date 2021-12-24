/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "CircleOperationExporterRule.h"
#include "CircleBuiltinTypesExtractor.h"
#include "Check.h"

#include <loco/IR/Graph.h>
#include <luci/IR/CircleNode.h>
#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleNodeVisitor.h>
#include <oops/InternalExn.h>

#include <vector>

namespace
{
class OutputVectorExtractor final : public luci::CircleNodeMutableVisitor<std::vector<int32_t>>
{
public:
  OutputVectorExtractor()
  {
    // DO NOTHING
  }

public:
  std::vector<int32_t> visit(luci::CircleNode *node) final
  {
    std::vector<int32_t> outputs_vec{luci::get_tensor_index(node)};
    return outputs_vec;
  }

  std::vector<int32_t> visit(luci::CircleBidirectionalSequenceLSTM *node) final
  {
    auto bidi_lstm_outs = loco::succs(node);
    assert((bidi_lstm_outs.size() == 1) || (bidi_lstm_outs.size() == 2));

    std::vector<int32_t> outputs_vec(bidi_lstm_outs.size());

    for (auto out : bidi_lstm_outs)
    {
      auto bidi_lstm_out = loco::must_cast<luci::CircleBidirectionalSequenceLSTMOut *>(out);
      if (bidi_lstm_out->index() >= int32_t(bidi_lstm_outs.size()))
        INTERNAL_EXN("Invalid BidirectionalSequenceLSTM output");
      outputs_vec[bidi_lstm_out->index()] = luci::get_tensor_index(bidi_lstm_out);
    }

    return outputs_vec;
  }

  std::vector<int32_t> visit(luci::CircleCustom *node) final
  {
    auto custom_outputs = loco::succs(node);
    assert(custom_outputs.size() == node->numOutputs());

    std::vector<int32_t> outputs_vec(node->numOutputs());

    for (auto out : custom_outputs)
    {
      auto custom_out = loco::must_cast<luci::CircleCustomOut *>(out);
      if (custom_out->index() >= int32_t(node->numOutputs()))
        INTERNAL_EXN("Invalid Custom output");
      outputs_vec[custom_out->index()] = luci::get_tensor_index(custom_out);
    }

    return outputs_vec;
  }

  std::vector<int32_t> visit(luci::CircleIf *node) final
  {
    auto if_outs = loco::succs(node);
    assert(if_outs.size() == node->output_count());

    std::vector<int32_t> outputs_vec(node->output_count());

    for (auto out : if_outs)
    {
      auto if_out = loco::must_cast<luci::CircleIfOut *>(out);
      if (if_out->index() >= int32_t(node->output_count()))
        INTERNAL_EXN("Invalid If output");
      outputs_vec[if_out->index()] = luci::get_tensor_index(if_out);
    }

    return outputs_vec;
  }

  std::vector<int32_t> visit(luci::CircleNonMaxSuppressionV4 *node) final
  {
    auto nms_outs = loco::succs(node);
    assert(nms_outs.size() == 2);

    std::vector<int32_t> outputs_vec(2);

    for (auto out : nms_outs)
    {
      auto nms_out = loco::must_cast<luci::CircleNonMaxSuppressionV4Out *>(out);
      if (nms_out->index() >= 2)
        INTERNAL_EXN("Invalid NonMaxSuppressionV4 output");
      outputs_vec[nms_out->index()] = luci::get_tensor_index(nms_out);
    }

    return outputs_vec;
  }

  std::vector<int32_t> visit(luci::CircleNonMaxSuppressionV5 *node) final
  {
    auto nms_outs = loco::succs(node);
    assert(nms_outs.size() == 3);

    std::vector<int32_t> outputs_vec(3);

    for (auto out : nms_outs)
    {
      auto nms_out = loco::must_cast<luci::CircleNonMaxSuppressionV5Out *>(out);
      if (nms_out->index() >= 3)
        INTERNAL_EXN("Invalid NonMaxSuppressionV5 output");
      outputs_vec[nms_out->index()] = luci::get_tensor_index(nms_out);
    }

    return outputs_vec;
  }

  std::vector<int32_t> visit(luci::CircleSplit *node) final
  {
    auto split_outs = loco::succs(node);
    assert(int32_t(split_outs.size()) == node->num_split());

    std::vector<int32_t> outputs_vec(node->num_split());

    for (auto out : split_outs)
    {
      auto split_out = loco::must_cast<luci::CircleSplitOut *>(out);
      if (split_out->index() >= node->num_split())
        INTERNAL_EXN("Invalid Split output");
      outputs_vec[split_out->index()] = luci::get_tensor_index(split_out);
    }

    return outputs_vec;
  }

  std::vector<int32_t> visit(luci::CircleSplitV *node) final
  {
    auto split_outs = loco::succs(node);
    assert(int32_t(split_outs.size()) == node->num_split());

    std::vector<int32_t> outputs_vec(node->num_split());

    for (auto out : split_outs)
    {
      auto split_out = loco::must_cast<luci::CircleSplitVOut *>(out);
      if (split_out->index() >= node->num_split())
        INTERNAL_EXN("Invalid SplitV output");
      outputs_vec[split_out->index()] = luci::get_tensor_index(split_out);
    }

    return outputs_vec;
  }

  std::vector<int32_t> visit(luci::CircleTopKV2 *node) final
  {
    auto topkv2_outs = loco::succs(node);
    assert(topkv2_outs.size() == 2);

    std::vector<int32_t> outputs_vec(2);

    for (auto out : topkv2_outs)
    {
      auto topkv2_out = loco::must_cast<luci::CircleTopKV2Out *>(out);
      if (topkv2_out->index() >= 2)
        INTERNAL_EXN("Invalid TopKV2 output");
      outputs_vec[topkv2_out->index()] = luci::get_tensor_index(topkv2_out);
    }

    return outputs_vec;
  }

  std::vector<int32_t> visit(luci::CircleUnique *node) final
  {
    auto unique_outs = loco::succs(node);
    assert(unique_outs.size() == 2);

    std::vector<int32_t> outputs_vec(2);

    for (auto out : unique_outs)
    {
      auto unique_out = loco::must_cast<luci::CircleUniqueOut *>(out);
      if (unique_out->index() >= 2)
        INTERNAL_EXN("Invalid Unique output");
      outputs_vec[unique_out->index()] = luci::get_tensor_index(unique_out);
    }

    return outputs_vec;
  }

  std::vector<int32_t> visit(luci::CircleUnpack *node) final
  {
    auto unpack_outs = loco::succs(node);
    assert(int32_t(unpack_outs.size()) == node->num());

    std::vector<int32_t> outputs_vec(node->num());

    for (auto out : unpack_outs)
    {
      auto unpack_out = loco::must_cast<luci::CircleUnpackOut *>(out);
      if (unpack_out->index() >= node->num())
        INTERNAL_EXN("Invalid Unpack output");
      outputs_vec[unpack_out->index()] = luci::get_tensor_index(unpack_out);
    }

    return outputs_vec;
  }

  std::vector<int32_t> visit(luci::CircleWhile *node) final
  {
    auto while_outs = loco::succs(node);
    assert(while_outs.size() == node->output_count());

    std::vector<int32_t> outputs_vec(node->output_count());

    for (auto out : while_outs)
    {
      auto while_out = loco::must_cast<luci::CircleWhileOut *>(out);
      if (while_out->index() >= int32_t(node->output_count()))
        INTERNAL_EXN("Invalid While output");
      outputs_vec[while_out->index()] = luci::get_tensor_index(while_out);
    }

    return outputs_vec;
  }
};

} // namespace

namespace luci
{

void OperationExporterRule::visit(luci::CircleNode *node)
{
  auto op_idx = _ctx.md.registerBuiltinOpcode(circle_builtin_operator(node),
                                              circle_custom_code(node), node->op_version());

  std::vector<int32_t> inputs_vec;
  for (uint32_t i = 0; i < node->arity(); ++i)
    inputs_vec.push_back(luci::get_tensor_index(node->arg(i)));
  auto inputs = _ctx.builder.CreateVector(inputs_vec);

  OutputVectorExtractor outputs_vec_extractor;
  auto outputs_vec = node->accept(&outputs_vec_extractor);
  auto outputs = _ctx.builder.CreateVector(outputs_vec);

  auto builtin_options = circle_builtin_options(node);

  luci::BuiltinOptionsExtractor builtin_options_extractor(_ctx.builder);
  auto options_offset = node->accept(&builtin_options_extractor);

  // If node is not CircleCustom, null offset(0) is returned
  auto custom_options = circle_custom_options(_ctx.builder, node);

  auto op_offset = circle::CreateOperator(_ctx.builder, op_idx, inputs, outputs, builtin_options,
                                          options_offset, custom_options);
  _ctx.gd._operators.push_back(op_offset);
}

} // namespace luci
