/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "CircleOperationExporter.h"
#include "CircleExporterUtils.h"
#include "Check.h"

#include <luci/IR/CircleNode.h>
#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleNodeVisitor.h>
#include <luci/Service/CircleShapeInference.h>
#include <luci/UserSettings.h>
#include <luci/Log.h>

#include <loco/IR/CanonicalNodeVisitor.h>
#include <oops/InternalExn.h>

#include <flatbuffers/flexbuffers.h>

using namespace flatbuffers;
using namespace circle;

namespace
{

using namespace luci;

struct ExportContext
{
  FlatBufferBuilder &builder;
  SerializedModelData &md;
  SerializedGraphData &gd;
};

/**
 * @brief Exports CircleMaxPool2D or CircleAveragePool2D
 *
 * @note  CirclePool2D should be one of CircleMaxPool2D or CircleAveragePool2D
 */
template <class CirclePool2D>
void export_pool_2d(ExportContext &ctx, CirclePool2D *node, circle::BuiltinOperator builtin_op)
{
  LUCI_ASSERT(builtin_op == circle::BuiltinOperator_MAX_POOL_2D ||
                  builtin_op == circle::BuiltinOperator_L2_POOL_2D ||
                  builtin_op == circle::BuiltinOperator_AVERAGE_POOL_2D,
              "Should be L2Pool, MaxPool or AvgPool");
  LUCI_ASSERT(node->padding() != luci::Padding::UNDEFINED, "Padding is not set");

  uint32_t op_idx = ctx.md.registerBuiltinOpcode(builtin_op, node->op_version());
  std::vector<int32_t> inputs_vec{get_tensor_index(node->value())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = ctx.builder.CreateVector(inputs_vec);
  auto outputs = ctx.builder.CreateVector(outputs_vec);

  circle::Padding padding = getOpPadding(node->padding());

  auto options = CreatePool2DOptions(ctx.builder, padding, node->stride()->w(), node->stride()->h(),
                                     node->filter()->w(), node->filter()->h(),
                                     to_circle_actfunc(node->fusedActivationFunction()));
  auto op_offset = CreateOperator(ctx.builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_Pool2DOptions, options.Union());
  ctx.gd._operators.push_back(op_offset);
}

/**
 * @brief export simple nodes
 */
void export_node(ExportContext &ctx, loco::Node *node, circle::BuiltinOperator bop,
                 circle::BuiltinOptions bot, flatbuffers::Offset<void> options_offset)
{
  uint32_t op_idx =
      ctx.md.registerBuiltinOpcode(bop, loco::must_cast<luci::CircleNode *>(node)->op_version());
  std::vector<int32_t> inputs_vec;
  std::vector<int32_t> outputs_vec{get_tensor_index(node)};
  for (uint32_t i = 0; i < node->arity(); ++i)
    inputs_vec.push_back(get_tensor_index(node->arg(i)));
  auto inputs = ctx.builder.CreateVector(inputs_vec);
  auto outputs = ctx.builder.CreateVector(outputs_vec);
  auto op_offset = CreateOperator(ctx.builder, op_idx, inputs, outputs, bot, options_offset);
  ctx.gd._operators.push_back(op_offset);
}

/**
 * @brief export simple nodes having void options
 */
void export_node(ExportContext &ctx, loco::Node *node, circle::BuiltinOperator bop)
{
  uint32_t op_idx =
      ctx.md.registerBuiltinOpcode(bop, loco::must_cast<luci::CircleNode *>(node)->op_version());
  std::vector<int32_t> inputs_vec;
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  for (uint32_t i = 0; i < node->arity(); ++i)
    inputs_vec.push_back(get_tensor_index(node->arg(i)));
  auto inputs = ctx.builder.CreateVector(inputs_vec);
  auto outputs = ctx.builder.CreateVector(outputs_vec);
  auto op_offset = CreateOperator(ctx.builder, op_idx, inputs, outputs);
  ctx.gd._operators.push_back(op_offset);
}

void export_node(ExportContext &ctx, luci::CircleAddN *node)
{
  uint32_t op_idx = ctx.md.registerBuiltinOpcode(circle::BuiltinOperator_ADD_N, node->op_version());
  std::vector<int32_t> inputs_vec;
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};

  for (uint32_t i = 0; i < node->arity(); ++i)
    inputs_vec.push_back(get_tensor_index(node->inputs(i)));

  auto inputs = ctx.builder.CreateVector(inputs_vec);
  auto outputs = ctx.builder.CreateVector(outputs_vec);
  auto options = CreateAddNOptions(ctx.builder);
  auto op_offset = CreateOperator(ctx.builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_AddNOptions, options.Union());
  ctx.gd._operators.push_back(op_offset);
}

void export_node(ExportContext &ctx, luci::CircleCast *node)
{
  uint32_t op_idx = ctx.md.registerBuiltinOpcode(circle::BuiltinOperator_CAST, node->op_version());
  std::vector<int32_t> inputs_vec{get_tensor_index(node->x())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = ctx.builder.CreateVector(inputs_vec);
  auto outputs = ctx.builder.CreateVector(outputs_vec);

  flatbuffers::Offset<Operator> op_offset;
  if (node->out_data_type() != loco::DataType::Unknown)
  {
    auto options = CreateCastOptions(ctx.builder, to_circle_tensortype(node->in_data_type()),
                                     to_circle_tensortype(node->out_data_type()));
    op_offset = CreateOperator(ctx.builder, op_idx, inputs, outputs,
                               circle::BuiltinOptions_CastOptions, options.Union());
  }
  else
  {
    op_offset = CreateOperator(ctx.builder, op_idx, inputs, outputs);
  }
  ctx.gd._operators.push_back(op_offset);
}

void export_node(ExportContext &ctx, luci::CircleConcatenation *node)
{
  uint32_t op_idx =
      ctx.md.registerBuiltinOpcode(circle::BuiltinOperator_CONCATENATION, node->op_version());
  std::vector<int32_t> inputs_vec;
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};

  for (uint32_t i = 0; i < node->numValues(); ++i)
    inputs_vec.push_back(get_tensor_index(node->values(i)));

  auto inputs = ctx.builder.CreateVector(inputs_vec);
  auto outputs = ctx.builder.CreateVector(outputs_vec);
  auto options = CreateConcatenationOptions(ctx.builder, node->axis(),
                                            to_circle_actfunc(node->fusedActivationFunction()));
  auto op_offset = CreateOperator(ctx.builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_ConcatenationOptions, options.Union());
  ctx.gd._operators.push_back(op_offset);
}

void export_node(ExportContext &ctx, luci::CircleCustom *node)
{
  auto custom_outputs = loco::succs(node);

  uint32_t op_idx = ctx.md.registerCustomOpcode(node->custom_code());
  std::vector<int32_t> inputs_vec;
  std::vector<int32_t> outputs_vec;

  for (uint32_t index = 0; index < node->numInputs(); index++)
  {
    inputs_vec.push_back(get_tensor_index(node->inputs(index)));
  }
  for (uint32_t index = 0; index < custom_outputs.size(); index++)
  {
    // store in order of index
    bool found = false;
    for (auto out : custom_outputs)
    {
      auto custom_out = loco::must_cast<luci::CircleCustomOut *>(out);
      if (custom_out->index() == static_cast<int32_t>(index))
      {
        outputs_vec.push_back(get_tensor_index(custom_out));
        found = true;
        break;
      }
    }
    if (!found)
    {
      INTERNAL_EXN("Invalid Custom output");
    }
  }

  auto inputs = ctx.builder.CreateVector(inputs_vec);
  auto outputs = ctx.builder.CreateVector(outputs_vec);
  flatbuffers::Offset<flatbuffers::Vector<uint8_t>> circle_custom_options;
  std::vector<uint8_t> custom_options_vec{node->custom_options().begin(),
                                          node->custom_options().end()};
  circle_custom_options = ctx.builder.CreateVector(custom_options_vec);
  auto op_offset = CreateOperator(ctx.builder, op_idx, inputs, outputs, circle::BuiltinOptions_NONE,
                                  flatbuffers::Offset<void>(), circle_custom_options);
  ctx.gd._operators.push_back(op_offset);
}

void export_node(ExportContext &ctx, luci::CircleIf *node)
{
  auto if_outs = loco::succs(node);
  assert(if_outs.size() == node->output_count());

  uint32_t op_idx = ctx.md.registerBuiltinOpcode(circle::BuiltinOperator_IF, node->op_version());
  std::vector<int32_t> inputs_vec;
  std::vector<int32_t> outputs_vec;

  inputs_vec.push_back(get_tensor_index(node->cond()));
  for (uint32_t idx = 0; idx < node->input_count(); ++idx)
    inputs_vec.push_back(get_tensor_index(node->input(idx)));

  for (uint32_t idx = 0; idx < node->output_count(); ++idx)
  {
    // store in order of index
    bool found = false;
    for (auto out : if_outs)
    {
      auto if_out = loco::must_cast<luci::CircleIfOut *>(out);
      if (if_out->index() == static_cast<int32_t>(idx))
      {
        outputs_vec.push_back(get_tensor_index(if_out));
        found = true;
        break;
      }
    }
    if (!found)
    {
      INTERNAL_EXN("Invalid CircleIf output");
    }
  }

  auto inputs = ctx.builder.CreateVector(inputs_vec);
  auto outputs = ctx.builder.CreateVector(outputs_vec);
  auto options = CreateIfOptions(ctx.builder, node->then_branch(), node->else_branch());
  auto op_offset = CreateOperator(ctx.builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_IfOptions, options.Union());
  ctx.gd._operators.push_back(op_offset);
}

void export_node(ExportContext &ctx, luci::CircleNonMaxSuppressionV4 *node)
{
  auto nms_outs = loco::succs(node);
  assert(nms_outs.size() == 2);

  uint32_t op_idx = ctx.md.registerBuiltinOpcode(circle::BuiltinOperator_NON_MAX_SUPPRESSION_V4,
                                                 node->op_version());
  std::vector<int32_t> inputs_vec{
      get_tensor_index(node->boxes()),           get_tensor_index(node->scores()),
      get_tensor_index(node->max_output_size()), get_tensor_index(node->iou_threshold()),
      get_tensor_index(node->score_threshold()),
  };
  std::vector<int32_t> outputs_vec;

  for (uint32_t idx = 0; idx < nms_outs.size(); ++idx)
  {
    // store in order of index
    bool found = false;
    for (auto out : nms_outs)
    {
      auto nms_out = loco::must_cast<luci::CircleNonMaxSuppressionV4Out *>(out);
      if (nms_out->index() == static_cast<int32_t>(idx))
      {
        outputs_vec.push_back(get_tensor_index(nms_out));
        found = true;
        break;
      }
    }
    if (!found)
    {
      INTERNAL_EXN("Invalid NonMaxSuppressionV4 output");
    }
  }

  auto inputs = ctx.builder.CreateVector(inputs_vec);
  auto outputs = ctx.builder.CreateVector(outputs_vec);
  auto options = CreateNonMaxSuppressionV4Options(ctx.builder);
  auto op_offset =
      CreateOperator(ctx.builder, op_idx, inputs, outputs,
                     circle::BuiltinOptions_NonMaxSuppressionV4Options, options.Union());
  ctx.gd._operators.push_back(op_offset);
}

void export_node(ExportContext &ctx, luci::CircleNonMaxSuppressionV5 *node)
{
  auto nms_outs = loco::succs(node);
  assert(nms_outs.size() == 3);

  uint32_t op_idx = ctx.md.registerBuiltinOpcode(circle::BuiltinOperator_NON_MAX_SUPPRESSION_V5,
                                                 node->op_version());
  std::vector<int32_t> inputs_vec{
      get_tensor_index(node->boxes()),           get_tensor_index(node->scores()),
      get_tensor_index(node->max_output_size()), get_tensor_index(node->iou_threshold()),
      get_tensor_index(node->score_threshold()), get_tensor_index(node->soft_nms_sigma()),
  };
  std::vector<int32_t> outputs_vec;

  for (uint32_t idx = 0; idx < nms_outs.size(); ++idx)
  {
    // store in order of index
    bool found = false;
    for (auto out : nms_outs)
    {
      auto nms_out = loco::must_cast<luci::CircleNonMaxSuppressionV5Out *>(out);
      if (nms_out->index() == static_cast<int32_t>(idx))
      {
        outputs_vec.push_back(get_tensor_index(nms_out));
        found = true;
        break;
      }
    }
    if (!found)
    {
      INTERNAL_EXN("Invalid NonMaxSuppressionV5 output");
    }
  }

  auto inputs = ctx.builder.CreateVector(inputs_vec);
  auto outputs = ctx.builder.CreateVector(outputs_vec);
  auto options = CreateNonMaxSuppressionV5Options(ctx.builder);
  auto op_offset =
      CreateOperator(ctx.builder, op_idx, inputs, outputs,
                     circle::BuiltinOptions_NonMaxSuppressionV5Options, options.Union());
  ctx.gd._operators.push_back(op_offset);
}

void export_node(ExportContext &ctx, luci::CircleReverseV2 *node)
{
  uint32_t op_idx =
      ctx.md.registerBuiltinOpcode(circle::BuiltinOperator_REVERSE_V2, node->op_version());
  std::vector<int32_t> inputs_vec{get_tensor_index(node->tensor()), get_tensor_index(node->axis())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = ctx.builder.CreateVector(inputs_vec);
  auto outputs = ctx.builder.CreateVector(outputs_vec);
  auto options = CreateReverseV2Options(ctx.builder);
  auto op_offset = CreateOperator(ctx.builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_ReverseSequenceOptions, options.Union());
  ctx.gd._operators.push_back(op_offset);
}

void export_node(ExportContext &ctx, luci::CircleSplit *node)
{
  auto split_outs = loco::succs(node);
  assert(int32_t(split_outs.size()) == node->num_split());

  uint32_t op_idx = ctx.md.registerBuiltinOpcode(circle::BuiltinOperator_SPLIT, node->op_version());
  // NOTE BuiltinOperator_SPLIT input is placed at second position
  std::vector<int32_t> inputs_vec{get_tensor_index(node->split_dim()),
                                  get_tensor_index(node->input())};
  std::vector<int32_t> outputs_vec;

  for (int32_t index = 0; index < node->num_split(); index++)
  {
    // store in order of index
    bool found = false;
    for (auto out : split_outs)
    {
      auto split_out = loco::must_cast<luci::CircleSplitOut *>(out);
      if (split_out->index() == index)
      {
        outputs_vec.push_back(get_tensor_index(split_out));
        found = true;
        break;
      }
    }
    if (!found)
    {
      INTERNAL_EXN("Invalid Split output");
    }
  }

  auto inputs = ctx.builder.CreateVector(inputs_vec);
  auto outputs = ctx.builder.CreateVector(outputs_vec);
  auto options = CreateSplitOptions(ctx.builder, node->num_split());
  auto op_offset = CreateOperator(ctx.builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_SplitOptions, options.Union());
  ctx.gd._operators.push_back(op_offset);
}

void export_node(ExportContext &ctx, luci::CircleSplitV *node)
{
  auto split_outs = loco::succs(node);
  assert(int32_t(split_outs.size()) == node->num_split());

  uint32_t op_idx =
      ctx.md.registerBuiltinOpcode(circle::BuiltinOperator_SPLIT_V, node->op_version());
  std::vector<int32_t> inputs_vec{get_tensor_index(node->input()),
                                  get_tensor_index(node->size_splits()),
                                  get_tensor_index(node->split_dim())};
  std::vector<int32_t> outputs_vec;

  for (int32_t index = 0; index < node->num_split(); index++)
  {
    // store in order of index
    bool found = false;
    for (auto out : split_outs)
    {
      auto split_out = loco::must_cast<luci::CircleSplitVOut *>(out);
      if (split_out->index() == index)
      {
        outputs_vec.push_back(get_tensor_index(split_out));
        found = true;
        break;
      }
    }
    if (!found)
    {
      INTERNAL_EXN("Invalid SplitV output");
    }
  }

  auto inputs = ctx.builder.CreateVector(inputs_vec);
  auto outputs = ctx.builder.CreateVector(outputs_vec);
  auto options = CreateSplitVOptions(ctx.builder, node->num_split());
  auto op_offset = CreateOperator(ctx.builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_SplitVOptions, options.Union());
  ctx.gd._operators.push_back(op_offset);
}

void export_node(ExportContext &ctx, luci::CircleTopKV2 *node)
{
  auto topkv2_outs = loco::succs(node);
  int outs_count = int32_t(topkv2_outs.size());
  assert(outs_count == 2);

  uint32_t op_idx =
      ctx.md.registerBuiltinOpcode(circle::BuiltinOperator_TOPK_V2, node->op_version());
  std::vector<int32_t> inputs_vec{get_tensor_index(node->input()), get_tensor_index(node->k())};
  std::vector<int32_t> outputs_vec;

  for (int32_t index = 0; index < outs_count; index++)
  {
    // store in order of index
    bool found = false;
    for (auto out : topkv2_outs)
    {
      auto topkv2_out = loco::must_cast<luci::CircleTopKV2Out *>(out);
      if (topkv2_out->index() == index)
      {
        outputs_vec.push_back(get_tensor_index(topkv2_out));
        found = true;
        break;
      }
    }
    if (!found)
    {
      INTERNAL_EXN("Invalid TopKV2 output");
    }
  }

  auto inputs = ctx.builder.CreateVector(inputs_vec);
  auto outputs = ctx.builder.CreateVector(outputs_vec);
  auto options = CreateTopKV2Options(ctx.builder);
  auto op_offset = CreateOperator(ctx.builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_TopKV2Options, options.Union());
  ctx.gd._operators.push_back(op_offset);
}

void export_node(ExportContext &ctx, luci::CircleUnique *node)
{
  auto unique_outs = loco::succs(node);
  assert(int32_t(unique_outs.size()) == 2);
  uint32_t op_idx =
      ctx.md.registerBuiltinOpcode(circle::BuiltinOperator_UNIQUE, node->op_version());

  std::vector<int32_t> inputs_vec{get_tensor_index(node->input())};
  std::vector<int32_t> outputs_vec;

  for (int32_t index = 0; index < 2; index++)
  {
    // store in order of index
    bool found = false;
    for (auto out : unique_outs)
    {
      auto unique_out = loco::must_cast<luci::CircleUniqueOut *>(out);
      if (unique_out->index() == index)
      {
        outputs_vec.push_back(get_tensor_index(unique_out));
        found = true;
        break;
      }
    }
    if (!found)
    {
      INTERNAL_EXN("Invalid Unique output");
    }
  }

  auto inputs = ctx.builder.CreateVector(inputs_vec);
  auto outputs = ctx.builder.CreateVector(outputs_vec);
  auto options = CreateUniqueOptions(ctx.builder, to_circle_tensortype(node->idx_out_type()));
  auto op_offset = CreateOperator(ctx.builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_UniqueOptions, options.Union());
  ctx.gd._operators.push_back(op_offset);
}

void export_node(ExportContext &ctx, luci::CircleUnpack *node)
{
  LOGGER(l);
  auto settings = luci::UserSettings::settings();

  auto unpack_outs = loco::succs(node);
  // NOTE real models may not use all of the outputs
  if (static_cast<int32_t>(unpack_outs.size()) != node->num())
  {
    if (settings->get(luci::UserSettings::Key::DisableValidation))
    {
      WARN(l) << "Warning: export Unpack(" << node->name() << ") 'num' not same as outputs";
    }
    else
      assert(false);
  }

  uint32_t op_idx =
      ctx.md.registerBuiltinOpcode(circle::BuiltinOperator_UNPACK, node->op_version());
  std::vector<int32_t> inputs_vec{get_tensor_index(node->value())};
  std::vector<int32_t> outputs_vec;

  for (int32_t index = 0; index < node->num(); index++)
  {
    // store in order of index
    bool found = false;
    for (auto out : unpack_outs)
    {
      auto unpack_out = loco::must_cast<luci::CircleUnpackOut *>(out);
      if (unpack_out->index() == index)
      {
        outputs_vec.push_back(get_tensor_index(unpack_out));
        found = true;
        break;
      }
    }
    // NOTE real models may not use all of the outputs
    if (!found)
    {
      if (settings->get(luci::UserSettings::Key::DisableValidation))
      {
        WARN(l) << "Warning: export Unpack(" << node->name() << ") output " << index << " not used";
      }
      else
        assert(false);
    }
  }

  auto inputs = ctx.builder.CreateVector(inputs_vec);
  auto outputs = ctx.builder.CreateVector(outputs_vec);
  auto options = CreateUnpackOptions(ctx.builder, node->num(), node->axis());
  auto op_offset = CreateOperator(ctx.builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_UnpackOptions, options.Union());
  ctx.gd._operators.push_back(op_offset);
}

void export_node(ExportContext &ctx, luci::CircleWhile *node)
{
  auto while_outs = loco::succs(node);
  assert(while_outs.size() == node->output_count());

  uint32_t op_idx = ctx.md.registerBuiltinOpcode(circle::BuiltinOperator_WHILE, node->op_version());
  std::vector<int32_t> inputs_vec;
  std::vector<int32_t> outputs_vec;

  for (uint32_t idx = 0; idx < node->input_count(); ++idx)
    inputs_vec.push_back(get_tensor_index(node->input(idx)));

  for (uint32_t idx = 0; idx < node->output_count(); ++idx)
  {
    // store in order of index
    bool found = false;
    for (auto out : while_outs)
    {
      auto while_out = loco::must_cast<luci::CircleWhileOut *>(out);
      if (while_out->index() == static_cast<int32_t>(idx))
      {
        outputs_vec.push_back(get_tensor_index(while_out));
        found = true;
        break;
      }
    }
    if (!found)
    {
      INTERNAL_EXN("Invalid CircleWhile output");
    }
  }

  auto inputs = ctx.builder.CreateVector(inputs_vec);
  auto outputs = ctx.builder.CreateVector(outputs_vec);
  auto options = CreateWhileOptions(ctx.builder, node->cond_branch(), node->body_branch());
  auto op_offset = CreateOperator(ctx.builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_WhileOptions, options.Union());
  ctx.gd._operators.push_back(op_offset);
}

class OperationExporter final : public luci::CircleNodeMutableVisitor<void>,
                                public loco::CanonicalNodeMutableVisitor<void>
{
public:
  OperationExporter(ExportContext &ctx) : _ctx{ctx}
  {
    // DO NOTHING
  }

public:
  void visit(luci::CircleAbs *) final;
  void visit(luci::CircleAdd *) final;
  void visit(luci::CircleAddN *) final;
  void visit(luci::CircleArgMax *) final;
  void visit(luci::CircleArgMin *) final;
  void visit(luci::CircleAveragePool2D *) final;
  void visit(luci::CircleBatchMatMul *) final;
  void visit(luci::CircleBatchToSpaceND *) final;
  void visit(luci::CircleBidirectionalSequenceLSTM *) final;
  void visit(luci::CircleCast *) final;
  void visit(luci::CircleCeil *) final;
  void visit(luci::CircleConcatenation *) final;
  void visit(luci::CircleConst *) final{/* skip, everything is done in exportOpDefinedTensors */};
  void visit(luci::CircleConv2D *) final;
  void visit(luci::CircleCos *) final;
  void visit(luci::CircleCustom *) final;
  void visit(luci::CircleDepthToSpace *) final;
  void visit(luci::CircleDepthwiseConv2D *) final;
  void visit(luci::CircleDequantize *) final;
  void visit(luci::CircleDiv *) final;
  void visit(luci::CircleElu *) final;
  void visit(luci::CircleEqual *) final;
  void visit(luci::CircleExp *) final;
  void visit(luci::CircleExpandDims *) final;
  void visit(luci::CircleFill *) final;
  void visit(luci::CircleFloor *) final;
  void visit(luci::CircleFloorDiv *) final;
  void visit(luci::CircleFloorMod *) final;
  void visit(luci::CircleFullyConnected *) final;
  void visit(luci::CircleGather *) final;
  void visit(luci::CircleGatherNd *) final;
  void visit(luci::CircleGreater *) final;
  void visit(luci::CircleGreaterEqual *) final;
  void visit(luci::CircleIf *) final;
  void visit(luci::CircleL2Normalize *) final;
  void visit(luci::CircleL2Pool2D *) final;
  void visit(luci::CircleLeakyRelu *) final;
  void visit(luci::CircleLess *) final;
  void visit(luci::CircleLessEqual *) final;
  void visit(luci::CircleLocalResponseNormalization *) final;
  void visit(luci::CircleLog *) final;
  void visit(luci::CircleLogicalAnd *) final;
  void visit(luci::CircleLogicalNot *) final;
  void visit(luci::CircleLogicalOr *) final;
  void visit(luci::CircleLogistic *) final;
  void visit(luci::CircleLogSoftmax *) final;
  void visit(luci::CircleMatrixDiag *) final;
  void visit(luci::CircleMatrixSetDiag *) final;
  void visit(luci::CircleMaximum *) final;
  void visit(luci::CircleMaxPool2D *) final;
  void visit(luci::CircleMean *) final;
  void visit(luci::CircleMinimum *) final;
  void visit(luci::CircleMirrorPad *) final;
  void visit(luci::CircleMul *) final;
  void visit(luci::CircleNeg *) final;
  void visit(luci::CircleNonMaxSuppressionV4 *) final;
  void visit(luci::CircleNonMaxSuppressionV5 *) final;
  void visit(luci::CircleNotEqual *) final;
  void visit(luci::CircleOneHot *) final;
  void visit(luci::CirclePack *) final;
  void visit(luci::CirclePad *) final;
  void visit(luci::CirclePadV2 *) final;
  void visit(luci::CirclePow *) final;
  void visit(luci::CirclePRelu *) final;
  void visit(luci::CircleRange *) final;
  void visit(luci::CircleRank *) final;
  void visit(luci::CircleReduceAny *) final;
  void visit(luci::CircleReduceMax *) final;
  void visit(luci::CircleReduceMin *) final;
  void visit(luci::CircleReduceProd *) final;
  void visit(luci::CircleRelu *) final;
  void visit(luci::CircleRelu6 *) final;
  void visit(luci::CircleReluN1To1 *) final;
  void visit(luci::CircleReshape *) final;
  void visit(luci::CircleResizeBilinear *) final;
  void visit(luci::CircleResizeNearestNeighbor *) final;
  void visit(luci::CircleReverseSequence *) final;
  void visit(luci::CircleReverseV2 *) final;
  void visit(luci::CircleRound *) final;
  void visit(luci::CircleRsqrt *) final;
  void visit(luci::CircleScatterNd *) final;
  void visit(luci::CircleSegmentSum *) final;
  void visit(luci::CircleSelect *) final;
  void visit(luci::CircleSelectV2 *) final;
  void visit(luci::CircleShape *) final;
  void visit(luci::CircleSin *) final;
  void visit(luci::CircleSlice *) final;
  void visit(luci::CircleSoftmax *) final;
  void visit(luci::CircleSpaceToBatchND *) final;
  void visit(luci::CircleSpaceToDepth *) final;
  void visit(luci::CircleSparseToDense *) final;
  void visit(luci::CircleSplit *) final;
  void visit(luci::CircleSplitV *) final;
  void visit(luci::CircleSqrt *) final;
  void visit(luci::CircleSquare *) final;
  void visit(luci::CircleSquaredDifference *) final;
  void visit(luci::CircleSqueeze *) final;
  void visit(luci::CircleStridedSlice *) final;
  void visit(luci::CircleSub *) final;
  void visit(luci::CircleSum *) final;
  void visit(luci::CircleTanh *) final;
  void visit(luci::CircleTile *) final;
  void visit(luci::CircleTopKV2 *) final;
  void visit(luci::CircleTranspose *) final;
  void visit(luci::CircleTransposeConv *) final;
  void visit(luci::CircleUnidirectionalSequenceLSTM *) final;
  void visit(luci::CircleUnique *) final;
  void visit(luci::CircleUnpack *) final;
  void visit(luci::CircleWhere *) final;
  void visit(luci::CircleWhile *) final;
  void visit(luci::CircleZerosLike *) final;
  // Circle only
  void visit(luci::CircleBCQFullyConnected *) final;
  void visit(luci::CircleBCQGather *) final;
  void visit(luci::CircleInstanceNorm *) final;
  // Virtual
  void visit(luci::CircleInput *) final {}
  void visit(luci::CircleOutput *) final {}
  void visit(luci::CircleOutputDummy *) final {}
  void visit(luci::CircleOutputExclude *) final {}
  // Virtual for multiple-outputs
  void visit(luci::CircleBidirectionalSequenceLSTMOut *) final {}
  void visit(luci::CircleCustomOut *) final {}
  void visit(luci::CircleIfOut *) final {}
  void visit(luci::CircleNonMaxSuppressionV4Out *) final {}
  void visit(luci::CircleNonMaxSuppressionV5Out *) final {}
  void visit(luci::CircleSplitOut *) final {}
  void visit(luci::CircleSplitVOut *) final {}
  void visit(luci::CircleTopKV2Out *) final {}
  void visit(luci::CircleUniqueOut *) final {}
  void visit(luci::CircleUnpackOut *) final {}
  void visit(luci::CircleWhileOut *) final {}

private:
  /**
   * @brief export simple nodes
   */
  void export_simple(loco::Node *node, circle::BuiltinOperator bop, circle::BuiltinOptions bot,
                     flatbuffers::Offset<void> options_offset);

  /**
   * @brief export simple nodes having void options
   */
  void export_simple(loco::Node *node, circle::BuiltinOperator bop);

private:
  ExportContext &_ctx;
};

void OperationExporter::export_simple(loco::Node *node, circle::BuiltinOperator bop,
                                      circle::BuiltinOptions bot,
                                      flatbuffers::Offset<void> options_offset)
{
  export_node(_ctx, node, bop, bot, options_offset);
}

void OperationExporter::export_simple(loco::Node *node, circle::BuiltinOperator bop)
{
  export_node(_ctx, node, bop);
}

void OperationExporter::visit(luci::CircleAbs *node)
{
  export_simple(node, circle::BuiltinOperator_ABS, circle::BuiltinOptions_AbsOptions,
                CreateAbsOptions(_ctx.builder).Union());
}

void OperationExporter::visit(luci::CircleAdd *node)
{
  export_simple(
      node, circle::BuiltinOperator_ADD, circle::BuiltinOptions_AddOptions,
      CreateAddOptions(_ctx.builder, to_circle_actfunc(node->fusedActivationFunction())).Union());
}

void OperationExporter::visit(luci::CircleAddN *node) { export_node(_ctx, node); }

void OperationExporter::visit(luci::CircleArgMax *node)
{
  export_simple(
      node, circle::BuiltinOperator_ARG_MAX, circle::BuiltinOptions_ArgMaxOptions,
      CreateArgMaxOptions(_ctx.builder, to_circle_tensortype(node->output_type())).Union());
}

void OperationExporter::visit(luci::CircleArgMin *node)
{
  export_simple(
      node, circle::BuiltinOperator_ARG_MIN, circle::BuiltinOptions_ArgMinOptions,
      CreateArgMinOptions(_ctx.builder, to_circle_tensortype(node->output_type())).Union());
}

void OperationExporter::visit(luci::CircleAveragePool2D *node)
{
  export_pool_2d<luci::CircleAveragePool2D>(_ctx, node, circle::BuiltinOperator_AVERAGE_POOL_2D);
}

void OperationExporter::visit(luci::CircleBatchMatMul *node)
{
  export_simple(node, circle::BuiltinOperator_BATCH_MATMUL,
                circle::BuiltinOptions_BatchMatMulOptions,
                CreateBatchMatMulOptions(_ctx.builder, node->adj_x(), node->adj_y()).Union());
}

void OperationExporter::visit(luci::CircleBidirectionalSequenceLSTM *node)
{
  auto bidi_lstm_outs = loco::succs(node);
  assert(int32_t(bidi_lstm_outs.size()) == 2);
  uint32_t op_idx = _ctx.md.registerBuiltinOpcode(
      circle::BuiltinOperator_BIDIRECTIONAL_SEQUENCE_LSTM, node->op_version());

  std::vector<int32_t> inputs_vec{get_tensor_index(node->input())};
  std::vector<int32_t> outputs_vec;

  for (int32_t index = 0; index < 2; index++)
  {
    // store in order of index
    bool found = false;
    for (auto out : bidi_lstm_outs)
    {
      auto bidi_lstm_out = loco::must_cast<luci::CircleBidirectionalSequenceLSTMOut *>(out);
      if (bidi_lstm_out->index() == index)
      {
        outputs_vec.push_back(get_tensor_index(bidi_lstm_out));
        found = true;
        break;
      }
    }
    if (!found)
    {
      INTERNAL_EXN("Invalid BidirectionalSequenceLSTM output");
    }
  }

  auto inputs = _ctx.builder.CreateVector(inputs_vec);
  auto outputs = _ctx.builder.CreateVector(outputs_vec);
  auto options = CreateBidirectionalSequenceLSTMOptions(
      _ctx.builder, to_circle_actfunc(node->fusedActivationFunction()), node->cell_clip(),
      node->proj_clip(), node->merge_outputs(), node->time_major(),
      node->asymmetric_quantize_inputs());
  auto op_offset =
      CreateOperator(_ctx.builder, op_idx, inputs, outputs,
                     circle::BuiltinOptions_BidirectionalSequenceLSTMOptions, options.Union());
  _ctx.gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleCast *node) { export_node(_ctx, node); }

void OperationExporter::visit(luci::CircleCeil *node)
{
  export_simple(node, circle::BuiltinOperator_CEIL);
}

void OperationExporter::visit(luci::CircleConcatenation *node) { export_node(_ctx, node); }

void OperationExporter::visit(luci::CircleBatchToSpaceND *node)
{
  export_simple(node, circle::BuiltinOperator_BATCH_TO_SPACE_ND,
                circle::BuiltinOptions_BatchToSpaceNDOptions,
                CreateBatchToSpaceNDOptions(_ctx.builder).Union());
}

void OperationExporter::visit(luci::CircleConv2D *node)
{
  export_simple(node, circle::BuiltinOperator_CONV_2D, circle::BuiltinOptions_Conv2DOptions,
                CreateConv2DOptions(_ctx.builder, getOpPadding(node->padding()),
                                    node->stride()->w(), node->stride()->h(),
                                    to_circle_actfunc(node->fusedActivationFunction()),
                                    node->dilation()->w(), node->dilation()->h())
                    .Union());
}

void OperationExporter::visit(luci::CircleCos *node)
{
  export_simple(node, circle::BuiltinOperator_COS, circle::BuiltinOptions_CosOptions,
                CreateCosOptions(_ctx.builder).Union());
}

void OperationExporter::visit(luci::CircleCustom *node) { export_node(_ctx, node); }

void OperationExporter::visit(luci::CircleDepthToSpace *node)
{
  export_simple(node, circle::BuiltinOperator_DEPTH_TO_SPACE,
                circle::BuiltinOptions_DepthToSpaceOptions,
                CreateDepthToSpaceOptions(_ctx.builder, node->block_size()).Union());
}

void OperationExporter::visit(luci::CircleDepthwiseConv2D *node)
{
  export_simple(node, circle::BuiltinOperator_DEPTHWISE_CONV_2D,
                circle::BuiltinOptions_DepthwiseConv2DOptions,
                CreateDepthwiseConv2DOptions(_ctx.builder, getOpPadding(node->padding()),
                                             node->stride()->w(), node->stride()->h(),
                                             node->depthMultiplier(),
                                             to_circle_actfunc(node->fusedActivationFunction()),
                                             node->dilation()->w(), node->dilation()->h())
                    .Union());
}

void OperationExporter::visit(luci::CircleDequantize *node)
{
  export_simple(node, circle::BuiltinOperator_DEQUANTIZE);
}

void OperationExporter::visit(luci::CircleDiv *node)
{
  export_simple(
      node, circle::BuiltinOperator_DIV, circle::BuiltinOptions_DivOptions,
      CreateDivOptions(_ctx.builder, to_circle_actfunc(node->fusedActivationFunction())).Union());
}

void OperationExporter::visit(luci::CircleElu *node)
{
  export_simple(node, circle::BuiltinOperator_ELU);
}

void OperationExporter::visit(luci::CircleEqual *node)
{
  export_simple(node, circle::BuiltinOperator_EQUAL, circle::BuiltinOptions_EqualOptions,
                CreateEqualOptions(_ctx.builder).Union());
}

void OperationExporter::visit(luci::CircleExp *node)
{
  export_simple(node, circle::BuiltinOperator_EXP, circle::BuiltinOptions_ExpOptions,
                CreateExpOptions(_ctx.builder).Union());
}

void OperationExporter::visit(luci::CircleExpandDims *node)
{
  export_simple(node, circle::BuiltinOperator_EXPAND_DIMS, circle::BuiltinOptions_ExpandDimsOptions,
                CreateExpandDimsOptions(_ctx.builder).Union());
}

void OperationExporter::visit(luci::CircleFill *node)
{
  export_simple(node, circle::BuiltinOperator_FILL, circle::BuiltinOptions_FillOptions,
                CreateFillOptions(_ctx.builder).Union());
}

void OperationExporter::visit(luci::CircleFloor *node)
{
  export_simple(node, circle::BuiltinOperator_FLOOR);
}

void OperationExporter::visit(luci::CircleFloorDiv *node)
{
  export_simple(node, circle::BuiltinOperator_FLOOR_DIV, circle::BuiltinOptions_FloorDivOptions,
                CreateFloorDivOptions(_ctx.builder).Union());
}

void OperationExporter::visit(luci::CircleFloorMod *node)
{
  export_simple(node, circle::BuiltinOperator_FLOOR_MOD, circle::BuiltinOptions_FloorModOptions,
                CreateFloorModOptions(_ctx.builder).Union());
}

void OperationExporter::visit(luci::CircleFullyConnected *node)
{
  export_simple(
      node, circle::BuiltinOperator_FULLY_CONNECTED, circle::BuiltinOptions_FullyConnectedOptions,
      CreateFullyConnectedOptions(_ctx.builder, to_circle_actfunc(node->fusedActivationFunction()),
                                  to_circle_weightsformat(node->weights_format()))
          .Union());
}

void OperationExporter::visit(luci::CircleGather *node)
{
  export_simple(node, circle::BuiltinOperator_GATHER, circle::BuiltinOptions_GatherOptions,
                CreateGatherOptions(_ctx.builder, node->axis()).Union());
}

void OperationExporter::visit(luci::CircleGatherNd *node)
{
  export_simple(node, circle::BuiltinOperator_GATHER_ND, circle::BuiltinOptions_GatherNdOptions,
                CreateGatherNdOptions(_ctx.builder).Union());
}

void OperationExporter::visit(luci::CircleGreater *node)
{
  export_simple(node, circle::BuiltinOperator_GREATER, circle::BuiltinOptions_GreaterOptions,
                CreateGreaterOptions(_ctx.builder).Union());
}

void OperationExporter::visit(luci::CircleGreaterEqual *node)
{
  export_simple(node, circle::BuiltinOperator_GREATER_EQUAL,
                circle::BuiltinOptions_GreaterEqualOptions,
                CreateGreaterEqualOptions(_ctx.builder).Union());
}

void OperationExporter::visit(luci::CircleIf *node) { export_node(_ctx, node); }

void OperationExporter::visit(luci::CircleL2Normalize *node)
{
  export_simple(
      node, circle::BuiltinOperator_L2_NORMALIZATION, circle::BuiltinOptions_L2NormOptions,
      CreateL2NormOptions(_ctx.builder, to_circle_actfunc(node->fusedActivationFunction()))
          .Union());
}

void OperationExporter::visit(luci::CircleL2Pool2D *node)
{
  export_pool_2d<luci::CircleL2Pool2D>(_ctx, node, circle::BuiltinOperator_L2_POOL_2D);
}

void OperationExporter::visit(luci::CircleLeakyRelu *node)
{
  export_simple(node, circle::BuiltinOperator_LEAKY_RELU, circle::BuiltinOptions_LeakyReluOptions,
                CreateLeakyReluOptions(_ctx.builder, node->alpha()).Union());
}

void OperationExporter::visit(luci::CircleLess *node)
{
  export_simple(node, circle::BuiltinOperator_LESS, circle::BuiltinOptions_LessOptions,
                CreateLessOptions(_ctx.builder).Union());
}

void OperationExporter::visit(luci::CircleLessEqual *node)
{
  export_simple(node, circle::BuiltinOperator_LESS_EQUAL, circle::BuiltinOptions_LessEqualOptions,
                CreateLessEqualOptions(_ctx.builder).Union());
}

void OperationExporter::visit(luci::CircleLocalResponseNormalization *node)
{
  export_simple(node, circle::BuiltinOperator_LOCAL_RESPONSE_NORMALIZATION,
                circle::BuiltinOptions_LocalResponseNormalizationOptions,
                CreateLocalResponseNormalizationOptions(_ctx.builder, node->radius(), node->bias(),
                                                        node->alpha(), node->beta())
                    .Union());
}

void OperationExporter::visit(luci::CircleLog *node)
{
  export_simple(node, circle::BuiltinOperator_LOG);
}

void OperationExporter::visit(luci::CircleLogicalAnd *node)
{
  export_simple(node, circle::BuiltinOperator_LOGICAL_AND, circle::BuiltinOptions_LogicalAndOptions,
                CreateLogicalAndOptions(_ctx.builder).Union());
}

void OperationExporter::visit(luci::CircleLogicalNot *node)
{
  export_simple(node, circle::BuiltinOperator_LOGICAL_NOT, circle::BuiltinOptions_LogicalNotOptions,
                CreateLogicalNotOptions(_ctx.builder).Union());
}

void OperationExporter::visit(luci::CircleLogicalOr *node)
{
  export_simple(node, circle::BuiltinOperator_LOGICAL_OR, circle::BuiltinOptions_LogicalOrOptions,
                CreateLogicalOrOptions(_ctx.builder).Union());
}

void OperationExporter::visit(luci::CircleLogistic *node)
{
  export_simple(node, circle::BuiltinOperator_LOGISTIC);
}

void OperationExporter::visit(luci::CircleLogSoftmax *node)
{
  export_simple(node, circle::BuiltinOperator_LOG_SOFTMAX, circle::BuiltinOptions_LogSoftmaxOptions,
                CreateLogSoftmaxOptions(_ctx.builder).Union());
}

void OperationExporter::visit(luci::CircleMatrixDiag *node)
{
  export_simple(node, circle::BuiltinOperator_MATRIX_DIAG, circle::BuiltinOptions_MatrixDiagOptions,
                CreateMatrixDiagOptions(_ctx.builder).Union());
}

void OperationExporter::visit(luci::CircleMatrixSetDiag *node)
{
  export_simple(node, circle::BuiltinOperator_MATRIX_SET_DIAG,
                circle::BuiltinOptions_MatrixSetDiagOptions,
                CreateMatrixSetDiagOptions(_ctx.builder).Union());
}

void OperationExporter::visit(luci::CircleMaximum *node)
{
  export_simple(node, circle::BuiltinOperator_MAXIMUM, circle::BuiltinOptions_MaximumMinimumOptions,
                CreateMaximumMinimumOptions(_ctx.builder).Union());
}

void OperationExporter::visit(luci::CircleMaxPool2D *node)
{
  export_pool_2d<luci::CircleMaxPool2D>(_ctx, node, circle::BuiltinOperator_MAX_POOL_2D);
}

void OperationExporter::visit(luci::CircleMean *node)
{
  export_simple(node, circle::BuiltinOperator_MEAN, circle::BuiltinOptions_ReducerOptions,
                CreateReducerOptions(_ctx.builder, node->keep_dims()).Union());
}

void OperationExporter::visit(luci::CircleMinimum *node)
{
  export_simple(node, circle::BuiltinOperator_MINIMUM, circle::BuiltinOptions_MaximumMinimumOptions,
                CreateMaximumMinimumOptions(_ctx.builder).Union());
}

void OperationExporter::visit(luci::CircleMirrorPad *node)
{
  export_simple(
      node, circle::BuiltinOperator_MIRROR_PAD, circle::BuiltinOptions_MirrorPadOptions,
      CreateMirrorPadOptions(_ctx.builder, to_circle_mirrorpadmode(node->mode())).Union());
}

void OperationExporter::visit(luci::CircleMul *node)
{
  export_simple(
      node, circle::BuiltinOperator_MUL, circle::BuiltinOptions_MulOptions,
      CreateMulOptions(_ctx.builder, to_circle_actfunc(node->fusedActivationFunction())).Union());
}

void OperationExporter::visit(luci::CircleNeg *node)
{
  export_simple(node, circle::BuiltinOperator_NEG, circle::BuiltinOptions_NegOptions,
                CreateNegOptions(_ctx.builder).Union());
}

void OperationExporter::visit(luci::CircleNonMaxSuppressionV4 *node) { export_node(_ctx, node); }

void OperationExporter::visit(luci::CircleNonMaxSuppressionV5 *node) { export_node(_ctx, node); }

void OperationExporter::visit(luci::CircleNotEqual *node)
{
  export_simple(node, circle::BuiltinOperator_NOT_EQUAL, circle::BuiltinOptions_NotEqualOptions,
                CreateNotEqualOptions(_ctx.builder).Union());
}

void OperationExporter::visit(luci::CircleOneHot *node)
{
  export_simple(node, circle::BuiltinOperator_ONE_HOT, circle::BuiltinOptions_OneHotOptions,
                CreateOneHotOptions(_ctx.builder, node->axis()).Union());
}

void OperationExporter::visit(luci::CirclePack *node)
{
  export_simple(node, circle::BuiltinOperator_PACK, circle::BuiltinOptions_PackOptions,
                CreatePackOptions(_ctx.builder, node->values_count(), node->axis()).Union());
}

void OperationExporter::visit(luci::CirclePad *node)
{
  export_simple(node, circle::BuiltinOperator_PAD, circle::BuiltinOptions_PadOptions,
                CreatePadOptions(_ctx.builder).Union());
}

void OperationExporter::visit(luci::CirclePadV2 *node)
{
  export_simple(node, circle::BuiltinOperator_PADV2, circle::BuiltinOptions_PadV2Options,
                CreatePadV2Options(_ctx.builder).Union());
}

void OperationExporter::visit(luci::CirclePow *node)
{
  export_simple(node, circle::BuiltinOperator_POW, circle::BuiltinOptions_PowOptions,
                CreatePowOptions(_ctx.builder).Union());
}

void OperationExporter::visit(luci::CirclePRelu *node)
{
  export_simple(node, circle::BuiltinOperator_PRELU);
}

void OperationExporter::visit(luci::CircleRange *node)
{
  export_simple(node, circle::BuiltinOperator_RANGE, circle::BuiltinOptions_RangeOptions,
                CreateRangeOptions(_ctx.builder).Union());
}

void OperationExporter::visit(luci::CircleRank *node)
{
  export_simple(node, circle::BuiltinOperator_RANK, circle::BuiltinOptions_RankOptions,
                CreateRankOptions(_ctx.builder).Union());
}

void OperationExporter::visit(luci::CircleReduceAny *node)
{
  export_simple(node, circle::BuiltinOperator_REDUCE_ANY, circle::BuiltinOptions_ReducerOptions,
                CreateReducerOptions(_ctx.builder, node->keep_dims()).Union());
}

void OperationExporter::visit(luci::CircleReduceMax *node)
{
  export_simple(node, circle::BuiltinOperator_REDUCE_MAX, circle::BuiltinOptions_ReducerOptions,
                CreateReducerOptions(_ctx.builder, node->keep_dims()).Union());
}

void OperationExporter::visit(luci::CircleReduceMin *node)
{
  export_simple(node, circle::BuiltinOperator_REDUCE_MIN, circle::BuiltinOptions_ReducerOptions,
                CreateReducerOptions(_ctx.builder, node->keep_dims()).Union());
}

void OperationExporter::visit(luci::CircleReduceProd *node)
{
  export_simple(node, circle::BuiltinOperator_REDUCE_PROD, circle::BuiltinOptions_ReducerOptions,
                CreateReducerOptions(_ctx.builder, node->keep_dims()).Union());
}

void OperationExporter::visit(luci::CircleRelu *node)
{
  export_simple(node, circle::BuiltinOperator_RELU);
}

void OperationExporter::visit(luci::CircleRelu6 *node)
{
  export_simple(node, circle::BuiltinOperator_RELU6);
}

void OperationExporter::visit(luci::CircleReluN1To1 *node)
{
  export_simple(node, circle::BuiltinOperator_RELU_N1_TO_1);
}

void OperationExporter::visit(luci::CircleReshape *node)
{
  auto new_shape = _ctx.builder.CreateVector<int32_t>(
      node->newShape()->rank(), [node](size_t i) { return node->newShape()->dim(i); });

  export_simple(node, circle::BuiltinOperator_RESHAPE, circle::BuiltinOptions_ReshapeOptions,
                CreateReshapeOptions(_ctx.builder, new_shape).Union());
}

void OperationExporter::visit(luci::CircleResizeBilinear *node)
{
  export_simple(
      node, circle::BuiltinOperator_RESIZE_BILINEAR, circle::BuiltinOptions_ResizeBilinearOptions,
      CreateResizeBilinearOptions(_ctx.builder, node->align_corners(), node->half_pixel_centers())
          .Union());
}

void OperationExporter::visit(luci::CircleResizeNearestNeighbor *node)
{
  export_simple(node, circle::BuiltinOperator_RESIZE_NEAREST_NEIGHBOR,
                circle::BuiltinOptions_ResizeNearestNeighborOptions,
                CreateResizeNearestNeighborOptions(_ctx.builder, node->align_corners()).Union());
}

void OperationExporter::visit(luci::CircleReverseSequence *node)
{
  export_simple(
      node, circle::BuiltinOperator_REVERSE_SEQUENCE, circle::BuiltinOptions_ReverseSequenceOptions,
      CreateReverseSequenceOptions(_ctx.builder, node->seq_axis(), node->batch_axis()).Union());
}

void OperationExporter::visit(luci::CircleReverseV2 *node) { export_node(_ctx, node); }

void OperationExporter::visit(luci::CircleRound *node)
{
  export_simple(node, circle::BuiltinOperator_ROUND);
}

void OperationExporter::visit(luci::CircleRsqrt *node)
{
  export_simple(node, circle::BuiltinOperator_RSQRT);
}

void OperationExporter::visit(luci::CircleScatterNd *node)
{
  export_simple(node, circle::BuiltinOperator_SCATTER_ND, circle::BuiltinOptions_ScatterNdOptions,
                CreateScatterNdOptions(_ctx.builder).Union());
}

void OperationExporter::visit(luci::CircleSegmentSum *node)
{
  export_simple(node, circle::BuiltinOperator_SEGMENT_SUM, circle::BuiltinOptions_SegmentSumOptions,
                CreateSegmentSumOptions(_ctx.builder).Union());
}

void OperationExporter::visit(luci::CircleSelect *node)
{
  export_simple(node, circle::BuiltinOperator_SELECT, circle::BuiltinOptions_SelectOptions,
                CreateSelectOptions(_ctx.builder).Union());
}

void OperationExporter::visit(luci::CircleSelectV2 *node)
{
  export_simple(node, circle::BuiltinOperator_SELECT_V2, circle::BuiltinOptions_SelectV2Options,
                CreateSelectV2Options(_ctx.builder).Union());
}

void OperationExporter::visit(luci::CircleShape *node)
{
  export_simple(node, circle::BuiltinOperator_SHAPE, circle::BuiltinOptions_ShapeOptions,
                CreateShapeOptions(_ctx.builder, to_circle_tensortype(node->out_type())).Union());
}

void OperationExporter::visit(luci::CircleSin *node)
{
  export_simple(node, circle::BuiltinOperator_SIN);
}

void OperationExporter::visit(luci::CircleSlice *node)
{
  export_simple(node, circle::BuiltinOperator_SLICE, circle::BuiltinOptions_SliceOptions,
                CreateSliceOptions(_ctx.builder).Union());
}

void OperationExporter::visit(luci::CircleSoftmax *node)
{
  export_simple(node, circle::BuiltinOperator_SOFTMAX, circle::BuiltinOptions_SoftmaxOptions,
                CreateSoftmaxOptions(_ctx.builder, node->beta()).Union());
}

void OperationExporter::visit(luci::CircleSpaceToBatchND *node)
{
  export_simple(node, circle::BuiltinOperator_SPACE_TO_BATCH_ND,
                circle::BuiltinOptions_SpaceToBatchNDOptions,
                CreateSpaceToBatchNDOptions(_ctx.builder).Union());
}

void OperationExporter::visit(luci::CircleSpaceToDepth *node)
{
  export_simple(node, circle::BuiltinOperator_SPACE_TO_DEPTH,
                circle::BuiltinOptions_SpaceToDepthOptions,
                CreateSpaceToDepthOptions(_ctx.builder, node->block_size()).Union());
}

void OperationExporter::visit(luci::CircleSparseToDense *node)
{
  export_simple(node, circle::BuiltinOperator_SPARSE_TO_DENSE,
                circle::BuiltinOptions_SparseToDenseOptions,
                CreateSparseToDenseOptions(_ctx.builder, node->validate_indices()).Union());
}

void OperationExporter::visit(luci::CircleSplit *node) { export_node(_ctx, node); }

void OperationExporter::visit(luci::CircleSplitV *node) { export_node(_ctx, node); }

void OperationExporter::visit(luci::CircleSqrt *node)
{
  export_simple(node, circle::BuiltinOperator_SQRT);
}

void OperationExporter::visit(luci::CircleSquare *node)
{
  export_simple(node, circle::BuiltinOperator_SQUARE, circle::BuiltinOptions_SquareOptions,
                CreateSquareOptions(_ctx.builder).Union());
}

void OperationExporter::visit(luci::CircleSquaredDifference *node)
{
  export_simple(node, circle::BuiltinOperator_SQUARED_DIFFERENCE,
                circle::BuiltinOptions_SquaredDifferenceOptions,
                CreateSquaredDifferenceOptions(_ctx.builder).Union());
}

void OperationExporter::visit(luci::CircleSqueeze *node)
{
  auto squeeze_dims = _ctx.builder.CreateVector<int32_t>(node->squeeze_dims());
  export_simple(node, circle::BuiltinOperator_SQUEEZE, circle::BuiltinOptions_SqueezeOptions,
                CreateSqueezeOptions(_ctx.builder, squeeze_dims).Union());
}

void OperationExporter::visit(luci::CircleStridedSlice *node)
{
  export_simple(node, circle::BuiltinOperator_STRIDED_SLICE,
                circle::BuiltinOptions_StridedSliceOptions,
                CreateStridedSliceOptions(_ctx.builder, node->begin_mask(), node->end_mask(),
                                          node->ellipsis_mask(), node->new_axis_mask(),
                                          node->shrink_axis_mask())
                    .Union());
}

void OperationExporter::visit(luci::CircleSub *node)
{
  export_simple(
      node, circle::BuiltinOperator_SUB, circle::BuiltinOptions_SubOptions,
      CreateSubOptions(_ctx.builder, to_circle_actfunc(node->fusedActivationFunction())).Union());
}

void OperationExporter::visit(luci::CircleSum *node)
{
  export_simple(node, circle::BuiltinOperator_SUM, circle::BuiltinOptions_ReducerOptions,
                CreateReducerOptions(_ctx.builder, node->keep_dims()).Union());
}

void OperationExporter::visit(luci::CircleTanh *node)
{
  export_simple(node, circle::BuiltinOperator_TANH);
}

void OperationExporter::visit(luci::CircleTile *node)
{
  export_simple(node, circle::BuiltinOperator_TILE, circle::BuiltinOptions_TileOptions,
                CreateTileOptions(_ctx.builder).Union());
}

void OperationExporter::visit(luci::CircleTopKV2 *node) { export_node(_ctx, node); }

void OperationExporter::visit(luci::CircleTranspose *node)
{
  export_simple(node, circle::BuiltinOperator_TRANSPOSE, circle::BuiltinOptions_TransposeOptions,
                CreateTransposeOptions(_ctx.builder).Union());
}

void OperationExporter::visit(luci::CircleTransposeConv *node)
{
  export_simple(node, circle::BuiltinOperator_TRANSPOSE_CONV,
                circle::BuiltinOptions_TransposeConvOptions,
                CreateTransposeConvOptions(_ctx.builder, getOpPadding(node->padding()),
                                           node->stride()->w(), node->stride()->h())
                    .Union());
}

void OperationExporter::visit(luci::CircleUnidirectionalSequenceLSTM *node)
{
  export_simple(node, circle::BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM,
                circle::BuiltinOptions_UnidirectionalSequenceLSTMOptions,
                CreateUnidirectionalSequenceLSTMOptions(
                    _ctx.builder, to_circle_actfunc(node->fusedActivationFunction()),
                    node->cell_clip(), node->proj_clip(), node->time_major(),
                    node->asymmetric_quantize_inputs())
                    .Union());
}

void OperationExporter::visit(luci::CircleUnique *node) { export_node(_ctx, node); }

void OperationExporter::visit(luci::CircleUnpack *node) { export_node(_ctx, node); }

void OperationExporter::visit(luci::CircleWhere *node)
{
  export_simple(node, circle::BuiltinOperator_WHERE, circle::BuiltinOptions_WhereOptions,
                CreateWhereOptions(_ctx.builder).Union());
}

void OperationExporter::visit(luci::CircleWhile *node) { export_node(_ctx, node); }

void OperationExporter::visit(luci::CircleZerosLike *node)
{
  export_simple(node, circle::BuiltinOperator_ZEROS_LIKE, circle::BuiltinOptions_ZerosLikeOptions,
                CreateZerosLikeOptions(_ctx.builder).Union());
}

void OperationExporter::visit(luci::CircleBCQFullyConnected *node)
{
  export_simple(node, circle::BuiltinOperator_BCQ_FULLY_CONNECTED,
                circle::BuiltinOptions_BCQFullyConnectedOptions,
                CreateBCQFullyConnectedOptions(_ctx.builder, node->weights_hidden_size(),
                                               to_circle_actfunc(node->fusedActivationFunction()))
                    .Union());
}

void OperationExporter::visit(luci::CircleBCQGather *node)
{
  export_simple(
      node, circle::BuiltinOperator_BCQ_GATHER, circle::BuiltinOptions_BCQGatherOptions,
      CreateBCQGatherOptions(_ctx.builder, node->input_hidden_size(), node->axis()).Union());
}

void OperationExporter::visit(luci::CircleInstanceNorm *node)
{
  export_simple(node, circle::BuiltinOperator_INSTANCE_NORM,
                circle::BuiltinOptions_InstanceNormOptions,
                CreateInstanceNormOptions(_ctx.builder, node->epsilon(),
                                          to_circle_actfunc(node->fusedActivationFunction()))
                    .Union());
}

void exportNode(loco::Node *node, flatbuffers::FlatBufferBuilder &builder, SerializedModelData &md,
                SerializedGraphData &gd)
{
  if (auto circle_node = dynamic_cast<luci::CircleNode *>(node))
  {
    ExportContext ctx{builder, md, gd};
    OperationExporter exporter{ctx};
    circle_node->accept(&exporter);
  }
  else
  {
    INTERNAL_EXN("Node with unsupported dialect found");
  }
}

} // namespace

namespace luci
{

void exportNodes(loco::Graph *g, FlatBufferBuilder &builder, SerializedModelData &md,
                 SerializedGraphData &gd)
{
  for (auto node : loco::postorder_traversal(loco::output_nodes(g)))
  {
    exportNode(node, builder, md, gd);
  }
}

} // namespace luci
