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

class OperationExporter final : public luci::CircleNodeMutableVisitor<void>,
                                public loco::CanonicalNodeMutableVisitor<void>
{
public:
  OperationExporter(FlatBufferBuilder &fbb, SerializedModelData &m, SerializedGraphData &g)
      : builder{fbb}, md{m}, gd{g}
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
  void visit(luci::CircleCast *) final;
  void visit(luci::CircleCeil *) final;
  void visit(luci::CircleConcatenation *) final;
  void visit(luci::CircleConst *) final{/* skip, everything is done in exportOpDefinedTensors */};
  void visit(luci::CircleConv2D *) final;
  void visit(luci::CircleCos *) final;
  void visit(luci::CircleCustom *) final;
  void visit(luci::CircleDepthToSpace *) final;
  void visit(luci::CircleDepthwiseConv2D *) final;
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
  void visit(luci::CircleMaximum *) final;
  void visit(luci::CircleMaxPool2D *) final;
  void visit(luci::CircleMean *) final;
  void visit(luci::CircleMinimum *) final;
  void visit(luci::CircleMirrorPad *) final;
  void visit(luci::CircleMul *) final;
  void visit(luci::CircleNeg *) final;
  void visit(luci::CircleNotEqual *) final;
  void visit(luci::CircleOneHot *) final;
  void visit(luci::CirclePack *) final;
  void visit(luci::CirclePad *) final;
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
  void visit(luci::CircleRound *) final;
  void visit(luci::CircleRsqrt *) final;
  void visit(luci::CircleScatterNd *) final;
  void visit(luci::CircleSelect *) final;
  void visit(luci::CircleShape *) final;
  void visit(luci::CircleSin *) final;
  void visit(luci::CircleSlice *) final;
  void visit(luci::CircleSoftmax *) final;
  void visit(luci::CircleSpaceToBatchND *) final;
  void visit(luci::CircleSpaceToDepth *) final;
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
  void visit(luci::CircleUnpack *) final;
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
  void visit(luci::CircleCustomOut *) final {}
  void visit(luci::CircleIfOut *) final {}
  void visit(luci::CircleSplitOut *) final {}
  void visit(luci::CircleSplitVOut *) final {}
  void visit(luci::CircleTopKV2Out *) final {}
  void visit(luci::CircleUnpackOut *) final {}
  void visit(luci::CircleWhileOut *) final {}

private:
  /**
   * @brief Exports CircleMaxPool2D or CircleAveragePool2D
   *
   * @note  CirclePool2D should be one of CircleMaxPool2D or CircleAveragePool2D
   */
  template <class CirclePool2D>
  void export_pool_2d(CirclePool2D *node, circle::BuiltinOperator builtin_op);

private:
  FlatBufferBuilder &builder;
  SerializedModelData &md;
  SerializedGraphData &gd;
};

template <class CirclePool2D>
void OperationExporter::export_pool_2d(CirclePool2D *node, circle::BuiltinOperator builtin_op)
{
  LUCI_ASSERT(builtin_op == circle::BuiltinOperator_MAX_POOL_2D ||
                  builtin_op == circle::BuiltinOperator_L2_POOL_2D ||
                  builtin_op == circle::BuiltinOperator_AVERAGE_POOL_2D,
              "Should be L2Pool, MaxPool or AvgPool");
  LUCI_ASSERT(node->padding() != luci::Padding::UNDEFINED, "Padding is not set");

  uint32_t op_idx = md.registerBuiltinOpcode(builtin_op);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->value())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);

  circle::Padding padding = getOpPadding(node->padding());

  auto options = CreatePool2DOptions(builder, padding, node->stride()->w(), node->stride()->h(),
                                     node->filter()->w(), node->filter()->h(),
                                     to_circle_actfunc(node->fusedActivationFunction()));
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_Pool2DOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleAbs *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_ABS);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->x())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateAbsOptions(builder);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_AbsOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleAdd *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_ADD);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->x()), get_tensor_index(node->y())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateAddOptions(builder, to_circle_actfunc(node->fusedActivationFunction()));
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_AddOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleAddN *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_ADD_N);
  std::vector<int32_t> inputs_vec;
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};

  for (uint32_t i = 0; i < node->arity(); ++i)
    inputs_vec.push_back(get_tensor_index(node->inputs(i)));

  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateAddNOptions(builder);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_AddNOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleArgMax *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_ARG_MAX);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->input()),
                                  get_tensor_index(node->dimension())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateArgMaxOptions(builder, to_circle_tensortype(node->output_type()));
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_ArgMaxOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleArgMin *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_ARG_MAX);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->input()),
                                  get_tensor_index(node->dimension())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateArgMinOptions(builder, to_circle_tensortype(node->output_type()));
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_ArgMinOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleAveragePool2D *node)
{
  export_pool_2d<luci::CircleAveragePool2D>(node, circle::BuiltinOperator_AVERAGE_POOL_2D);
}

void OperationExporter::visit(luci::CircleBatchMatMul *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_BATCH_MATMUL);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->x()), get_tensor_index(node->y())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateBatchMatMulOptions(builder, node->adj_x(), node->adj_y());
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_BatchMatMulOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleCast *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_CAST);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->x())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);

  flatbuffers::Offset<Operator> op_offset;
  if (node->out_data_type() != loco::DataType::Unknown)
  {
    auto options = CreateCastOptions(builder, to_circle_tensortype(node->in_data_type()),
                                     to_circle_tensortype(node->out_data_type()));
    op_offset = CreateOperator(builder, op_idx, inputs, outputs, circle::BuiltinOptions_CastOptions,
                               options.Union());
  }
  else
  {
    op_offset = CreateOperator(builder, op_idx, inputs, outputs);
  }
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleCeil *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_CEIL);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->x())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs);
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleConcatenation *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_CONCATENATION);
  std::vector<int32_t> inputs_vec;
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};

  for (uint32_t i = 0; i < node->numValues(); ++i)
    inputs_vec.push_back(get_tensor_index(node->values(i)));

  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateConcatenationOptions(builder, node->axis(),
                                            to_circle_actfunc(node->fusedActivationFunction()));
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_ConcatenationOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleBatchToSpaceND *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_BATCH_TO_SPACE_ND);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->input()),
                                  get_tensor_index(node->block_shape()),
                                  get_tensor_index(node->crops())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};

  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateBatchToSpaceNDOptions(builder);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_BatchToSpaceNDOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleConv2D *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_CONV_2D);

  // Make input, output and options for operator
  std::vector<int32_t> inputs_vec{get_tensor_index(node->input()), get_tensor_index(node->filter()),
                                  get_tensor_index(node->bias())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  circle::Padding padding = getOpPadding(node->padding());
  auto options = CreateConv2DOptions(builder, padding, node->stride()->w(), node->stride()->h(),
                                     to_circle_actfunc(node->fusedActivationFunction()),
                                     node->dilation()->w(), node->dilation()->h());

  // Make CONV_2D operator
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_Conv2DOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleCos *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_COS);

  // Make input, output and options for operator
  std::vector<int32_t> inputs_vec{get_tensor_index(node->x())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateCosOptions(builder);

  // Make COS operator
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_CosOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleCustom *node)
{
  auto custom_outputs = loco::succs(node);

  uint32_t op_idx = md.registerCustomOpcode(node->custom_code());
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

  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  flatbuffers::Offset<flatbuffers::Vector<uint8_t>> circle_custom_options;
  std::vector<uint8_t> custom_options_vec{node->custom_options().begin(),
                                          node->custom_options().end()};
  circle_custom_options = builder.CreateVector(custom_options_vec);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs, circle::BuiltinOptions_NONE,
                                  flatbuffers::Offset<void>(), circle_custom_options);
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleDepthToSpace *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_DEPTH_TO_SPACE);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->input())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};

  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateDepthToSpaceOptions(builder, node->block_size());
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_DepthToSpaceOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleDepthwiseConv2D *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_DEPTHWISE_CONV_2D);

  // Make input, output and options for operator
  std::vector<int32_t> inputs_vec{get_tensor_index(node->input()), get_tensor_index(node->filter()),
                                  get_tensor_index(node->bias())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  circle::Padding padding = getOpPadding(node->padding());
  auto options = CreateDepthwiseConv2DOptions(builder, padding, node->stride()->w(),
                                              node->stride()->h(), node->depthMultiplier(),
                                              to_circle_actfunc(node->fusedActivationFunction()),
                                              node->dilation()->w(), node->dilation()->h());

  // Make DEPTHWISE_CONV_2D operator
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_DepthwiseConv2DOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleDiv *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_DIV);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->x()), get_tensor_index(node->y())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateDivOptions(builder, to_circle_actfunc(node->fusedActivationFunction()));
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_DivOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleElu *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_ELU);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->features())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs);
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleEqual *node)
{
  uint32_t opcode_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_EQUAL);
  std::vector<int32_t> inputs{get_tensor_index(node->x()), get_tensor_index(node->y())};
  std::vector<int32_t> outputs{get_tensor_index(node)};

  auto fb_inputs = builder.CreateVector(inputs);
  auto fb_outputs = builder.CreateVector(outputs);

  auto options = CreateEqualOptions(builder);

  auto op_offset = CreateOperator(builder, opcode_idx, fb_inputs, fb_outputs,
                                  circle::BuiltinOptions_EqualOptions, options.Union());

  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleExp *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_EXP);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->x())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateExpOptions(builder);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_ExpOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleExpandDims *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_EXPAND_DIMS);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->input()), get_tensor_index(node->axis())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateExpOptions(builder);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_ExpOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleFill *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_FILL);

  // Create inputs and outputs.
  std::vector<int32_t> inputs_vec{get_tensor_index(node->dims()), get_tensor_index(node->value())};
  std::vector<int32_t> outputs_vec{get_tensor_index(node)};

  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);

  // Create options.
  auto options = CreateFillOptions(builder);

  // Create the operator.
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_FillOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleFloor *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_FLOOR);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->x())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs);
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleFloorDiv *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_FLOOR_DIV);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->x()), get_tensor_index(node->y())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateFloorDivOptions(builder);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_FloorDivOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleFloorMod *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_FLOOR_MOD);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->x()), get_tensor_index(node->y())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateFloorModOptions(builder);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_FloorModOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleFullyConnected *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_FULLY_CONNECTED);

  // Make input, output and options for operator
  std::vector<int32_t> inputs_vec{get_tensor_index(node->input()),
                                  get_tensor_index(node->weights()),
                                  get_tensor_index(node->bias())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options =
      CreateFullyConnectedOptions(builder, to_circle_actfunc(node->fusedActivationFunction()));

  // Make FULLY_CONNECTED operator
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_FullyConnectedOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleGather *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_GATHER);

  // Make input, output and options for operator
  std::vector<int32_t> inputs_vec{get_tensor_index(node->params()),
                                  get_tensor_index(node->indices())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateGatherOptions(builder, node->axis());

  // Make GATHER operator
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_GatherOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleGatherNd *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_GATHER_ND);

  // Make input, output and options for operator
  std::vector<int32_t> inputs_vec{get_tensor_index(node->params()),
                                  get_tensor_index(node->indices())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateGatherNdOptions(builder);

  // Make GATHER_ND operator
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_GatherNdOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleGreater *node)
{
  uint32_t opcode_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_GREATER);
  std::vector<int32_t> inputs{get_tensor_index(node->x()), get_tensor_index(node->y())};
  std::vector<int32_t> outputs{get_tensor_index(node)};

  auto fb_inputs = builder.CreateVector(inputs);
  auto fb_outputs = builder.CreateVector(outputs);

  auto options = CreateGreaterOptions(builder);

  auto op_offset = CreateOperator(builder, opcode_idx, fb_inputs, fb_outputs,
                                  circle::BuiltinOptions_GreaterOptions, options.Union());

  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleGreaterEqual *node)
{
  uint32_t opcode_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_GREATER_EQUAL);
  std::vector<int32_t> inputs{get_tensor_index(node->x()), get_tensor_index(node->y())};
  std::vector<int32_t> outputs{get_tensor_index(node)};

  auto fb_inputs = builder.CreateVector(inputs);
  auto fb_outputs = builder.CreateVector(outputs);

  auto options = CreateGreaterEqualOptions(builder);

  auto op_offset = CreateOperator(builder, opcode_idx, fb_inputs, fb_outputs,
                                  circle::BuiltinOptions_GreaterEqualOptions, options.Union());

  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleIf *node)
{
  auto if_outs = loco::succs(node);
  assert(if_outs.size() == node->output_count());

  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_IF);
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

  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateIfOptions(builder, node->then_branch(), node->else_branch());
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_IfOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleL2Normalize *node)
{
  uint32_t opcode_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_L2_NORMALIZATION);
  std::vector<int32_t> inputs{get_tensor_index(node->x())};
  std::vector<int32_t> outputs{get_tensor_index(node)};

  auto fb_inputs = builder.CreateVector(inputs);
  auto fb_outputs = builder.CreateVector(outputs);

  auto options = CreateL2NormOptions(builder, to_circle_actfunc(node->fusedActivationFunction()));

  auto op_offset = CreateOperator(builder, opcode_idx, fb_inputs, fb_outputs,
                                  circle::BuiltinOptions_L2NormOptions, options.Union());

  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleL2Pool2D *node)
{
  export_pool_2d<luci::CircleL2Pool2D>(node, circle::BuiltinOperator_L2_POOL_2D);
}

void OperationExporter::visit(luci::CircleLeakyRelu *node)
{
  uint32_t opcode_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_LEAKY_RELU);
  std::vector<int32_t> inputs{get_tensor_index(node->features())};
  std::vector<int32_t> outputs{get_tensor_index(node)};

  auto fb_inputs = builder.CreateVector(inputs);
  auto fb_outputs = builder.CreateVector(outputs);

  auto options = CreateLeakyReluOptions(builder, node->alpha());

  auto op_offset = CreateOperator(builder, opcode_idx, fb_inputs, fb_outputs,
                                  circle::BuiltinOptions_LeakyReluOptions, options.Union());

  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleLess *node)
{
  uint32_t opcode_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_LESS);
  std::vector<int32_t> inputs{get_tensor_index(node->x()), get_tensor_index(node->y())};
  std::vector<int32_t> outputs{get_tensor_index(node)};

  auto fb_inputs = builder.CreateVector(inputs);
  auto fb_outputs = builder.CreateVector(outputs);

  auto options = CreateLessOptions(builder);

  auto op_offset = CreateOperator(builder, opcode_idx, fb_inputs, fb_outputs,
                                  circle::BuiltinOptions_LessOptions, options.Union());

  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleLessEqual *node)
{
  uint32_t opcode_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_LESS_EQUAL);
  std::vector<int32_t> inputs{get_tensor_index(node->x()), get_tensor_index(node->y())};
  std::vector<int32_t> outputs{get_tensor_index(node)};

  auto fb_inputs = builder.CreateVector(inputs);
  auto fb_outputs = builder.CreateVector(outputs);

  auto options = CreateLessEqualOptions(builder);

  auto op_offset = CreateOperator(builder, opcode_idx, fb_inputs, fb_outputs,
                                  circle::BuiltinOptions_LessEqualOptions, options.Union());

  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleLocalResponseNormalization *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_LOCAL_RESPONSE_NORMALIZATION);

  // Make input, output and options for operator
  std::vector<int32_t> inputs_vec{get_tensor_index(node->input())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateLocalResponseNormalizationOptions(builder);

  // Make LOCAL_RESPONSE_NORMALIZATION operator
  auto op_offset =
      CreateOperator(builder, op_idx, inputs, outputs,
                     circle::BuiltinOptions_LocalResponseNormalizationOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleLog *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_LOG);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->x())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  // Make LOG operator; LOG does not have Options
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs);
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleLogicalAnd *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_LOGICAL_AND);

  // Make input, output and options for operator
  std::vector<int32_t> inputs_vec{get_tensor_index(node->x()), get_tensor_index(node->y())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateLogicalAndOptions(builder);

  // Make LOGICAL_AND operator
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_LogicalAndOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleLogicalNot *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_LOGICAL_NOT);

  // Make input, output and options for operator
  std::vector<int32_t> inputs_vec{get_tensor_index(node->x())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateLogicalNotOptions(builder);

  // Make LOGICAL_NOT operator
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_LogicalNotOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleLogicalOr *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_LOGICAL_OR);

  // Make input, output and options for operator
  std::vector<int32_t> inputs_vec{get_tensor_index(node->x()), get_tensor_index(node->y())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateLogicalOrOptions(builder);

  // Make LOGICAL_OR operator
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_LogicalOrOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleLogistic *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_LOGISTIC);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->x())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs);
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleLogSoftmax *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_LOG_SOFTMAX);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->logits())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateLogSoftmaxOptions(builder);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_LogSoftmaxOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleMatrixDiag *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_MATRIX_DIAG);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->diagonal())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateMatrixDiagOptions(builder);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_MatrixDiagOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleMaximum *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_MAXIMUM);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->x()), get_tensor_index(node->y())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateMaximumMinimumOptions(builder);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_MaximumMinimumOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleMaxPool2D *node)
{
  export_pool_2d<luci::CircleMaxPool2D>(node, circle::BuiltinOperator_MAX_POOL_2D);
}

void OperationExporter::visit(luci::CircleMean *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_MEAN);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->input()),
                                  get_tensor_index(node->reduction_indices())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateReducerOptions(builder, node->keep_dims());
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_ReducerOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleMinimum *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_MINIMUM);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->x()), get_tensor_index(node->y())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateMaximumMinimumOptions(builder);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_MaximumMinimumOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleMirrorPad *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_MIRROR_PAD);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->input()),
                                  get_tensor_index(node->paddings())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateMirrorPadOptions(builder, to_circle_mirrorpadmode(node->mode()));
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_MirrorPadOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleMul *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_MUL);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->x()), get_tensor_index(node->y())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateMulOptions(builder, to_circle_actfunc(node->fusedActivationFunction()));
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_MulOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleNeg *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_NEG);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->x())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateNegOptions(builder);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_NegOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleNotEqual *node)
{
  uint32_t opcode_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_NOT_EQUAL);
  std::vector<int32_t> inputs{get_tensor_index(node->x()), get_tensor_index(node->y())};
  std::vector<int32_t> outputs{get_tensor_index(node)};

  auto fb_inputs = builder.CreateVector(inputs);
  auto fb_outputs = builder.CreateVector(outputs);

  auto options = CreateNotEqualOptions(builder);

  auto op_offset = CreateOperator(builder, opcode_idx, fb_inputs, fb_outputs,
                                  circle::BuiltinOptions_NotEqualOptions, options.Union());

  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleOneHot *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_ONE_HOT);
  std::vector<int32_t> inputs_vec{
      get_tensor_index(node->indices()), get_tensor_index(node->depth()),
      get_tensor_index(node->on_value()), get_tensor_index(node->off_value())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateOneHotOptions(builder, node->axis());

  // Make ONEHOT operator
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_OneHotOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CirclePack *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_PACK);
  std::vector<int32_t> inputs_vec;
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};

  for (uint32_t i = 0; i < node->values_count(); ++i)
    inputs_vec.push_back(get_tensor_index(node->values(i)));

  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreatePackOptions(builder, node->values_count(), node->axis());
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_PackOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CirclePad *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_PAD);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->input()),
                                  get_tensor_index(node->paddings())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreatePadOptions(builder);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_PadOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CirclePow *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_POW);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->x()), get_tensor_index(node->y())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreatePowOptions(builder);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_PowOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CirclePRelu *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_PRELU);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->input()), get_tensor_index(node->alpha())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  // Make PRelu operator; PRelu does not have Options
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs);
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleRange *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_RANGE);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->start()), get_tensor_index(node->limit()),
                                  get_tensor_index(node->delta())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateRangeOptions(builder);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_RangeOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleRank *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_RANK);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->input())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateRankOptions(builder);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_RankOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleReduceAny *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_REDUCE_ANY);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->input()),
                                  get_tensor_index(node->reduction_indices())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateReducerOptions(builder, node->keep_dims());
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_ReducerOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleReduceMax *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_REDUCE_MAX);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->input()),
                                  get_tensor_index(node->reduction_indices())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateReducerOptions(builder, node->keep_dims());
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_ReducerOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleReduceMin *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_REDUCE_MIN);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->input()),
                                  get_tensor_index(node->reduction_indices())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateReducerOptions(builder, node->keep_dims());
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_ReducerOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleReduceProd *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_REDUCE_PROD);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->input()),
                                  get_tensor_index(node->reduction_indices())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateReducerOptions(builder, node->keep_dims());
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_ReducerOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleRelu *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_RELU);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->features())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs);
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleRelu6 *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_RELU6);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->features())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs);
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleReluN1To1 *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_RELU_N1_TO_1);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->features())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs);
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleReshape *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_RESHAPE);

  // Create inputs and outputs.
  std::vector<int32_t> inputs_vec{get_tensor_index(node->tensor()),
                                  get_tensor_index(node->shape())};
  std::vector<int32_t> outputs_vec{get_tensor_index(node)};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);

  // Create options.
  auto new_shape = builder.CreateVector<int32_t>(
      node->newShape()->rank(), [node](size_t i) { return node->newShape()->dim(i); });
  auto options = CreateReshapeOptions(builder, new_shape);

  // Create the operator.
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_ReshapeOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleResizeBilinear *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_RESIZE_BILINEAR);

  // Create inputs and outputs.
  std::vector<int32_t> inputs_vec{get_tensor_index(node->input()), get_tensor_index(node->size())};

  std::vector<int32_t> outputs_vec{get_tensor_index(node)};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);

  // Create options.
  auto options =
      CreateResizeBilinearOptions(builder, node->align_corners(), node->half_pixel_centers());

  // Create the operator.
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_ResizeBilinearOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleResizeNearestNeighbor *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_RESIZE_NEAREST_NEIGHBOR);

  // Create inputs and outputs.
  std::vector<int32_t> inputs_vec{get_tensor_index(node->input()), get_tensor_index(node->size())};

  std::vector<int32_t> outputs_vec{get_tensor_index(node)};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);

  // Create options.
  auto options = CreateResizeNearestNeighborOptions(builder, node->align_corners());

  // Create the operator.
  auto op_offset =
      CreateOperator(builder, op_idx, inputs, outputs,
                     circle::BuiltinOptions_ResizeNearestNeighborOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleReverseSequence *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_REVERSE_SEQUENCE);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->input()),
                                  get_tensor_index(node->seq_lengths())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateReverseSequenceOptions(builder, node->seq_axis(), node->batch_axis());
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_ReverseSequenceOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleRound *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_ROUND);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->x())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs);
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleRsqrt *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_RSQRT);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->x())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs);
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleScatterNd *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_SCATTER_ND);

  // Make input, output and options for operator
  std::vector<int32_t> inputs_vec{get_tensor_index(node->indices()),
                                  get_tensor_index(node->updates()),
                                  get_tensor_index(node->shape())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateScatterNdOptions(builder);

  // Make SCATTER_ND operator
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_ScatterNdOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleSelect *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_SELECT);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->condition()), get_tensor_index(node->t()),
                                  get_tensor_index(node->e())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateSelectOptions(builder);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_SelectOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleShape *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_SHAPE);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->input())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateShapeOptions(builder, to_circle_tensortype(node->out_type()));
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_ShapeOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleSin *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_SIN);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->x())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  // Make SIN operator; SIN does not have Options
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs);
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleSlice *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_SLICE);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->input()), get_tensor_index(node->begin()),
                                  get_tensor_index(node->size())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateSliceOptions(builder);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_SliceOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleSoftmax *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_SOFTMAX);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->logits())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateSoftmaxOptions(builder, node->beta());
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_SoftmaxOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleSpaceToBatchND *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_SPACE_TO_BATCH_ND);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->input()),
                                  get_tensor_index(node->block_shape()),
                                  get_tensor_index(node->paddings())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};

  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateSpaceToBatchNDOptions(builder);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_SpaceToBatchNDOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleSpaceToDepth *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_SPACE_TO_DEPTH);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->input())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};

  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateSpaceToDepthOptions(builder);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_SpaceToDepthOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleSplit *node)
{
  auto split_outs = loco::succs(node);
  assert(int32_t(split_outs.size()) == node->num_split());

  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_SPLIT);
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

  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateSplitOptions(builder, node->num_split());
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_SplitOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleSplitV *node)
{
  auto split_outs = loco::succs(node);
  assert(int32_t(split_outs.size()) == node->num_split());

  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_SPLIT_V);
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

  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateSplitVOptions(builder, node->num_split());
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_SplitVOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleSqrt *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_SQRT);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->x())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs);
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleSquare *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_SQUARE);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->x())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateSquareOptions(builder);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_SquareOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleSquaredDifference *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_SQUARED_DIFFERENCE);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->x()), get_tensor_index(node->y())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateSquaredDifferenceOptions(builder);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_SquaredDifferenceOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleSqueeze *node)
{
  uint32_t opcode_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_SQUEEZE);
  std::vector<int32_t> inputs{get_tensor_index(node->input())};
  std::vector<int32_t> outputs{get_tensor_index(node)};

  auto fb_inputs = builder.CreateVector(inputs);
  auto fb_outputs = builder.CreateVector(outputs);

  auto squeeze_dims = builder.CreateVector<int32_t>(node->squeeze_dims());

  auto options = CreateSqueezeOptions(builder, squeeze_dims);
  auto op_offset = CreateOperator(builder, opcode_idx, fb_inputs, fb_outputs,
                                  circle::BuiltinOptions_SqueezeOptions, options.Union());

  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleStridedSlice *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_STRIDED_SLICE);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->input()), get_tensor_index(node->begin()),
                                  get_tensor_index(node->end()), get_tensor_index(node->strides())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateStridedSliceOptions(builder, node->begin_mask(), node->end_mask(),
                                           node->ellipsis_mask(), node->new_axis_mask(),
                                           node->shrink_axis_mask());
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_StridedSliceOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleSub *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_SUB);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->x()), get_tensor_index(node->y())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateSubOptions(builder, to_circle_actfunc(node->fusedActivationFunction()));
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_SubOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleSum *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_SUM);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->input()),
                                  get_tensor_index(node->reduction_indices())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateReducerOptions(builder, node->keep_dims());
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_ReducerOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleTanh *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_TANH);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->x())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs);
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleTile *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_TILE);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->input()),
                                  get_tensor_index(node->multiples())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateTileOptions(builder);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_TileOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleTopKV2 *node)
{
  auto topkv2_outs = loco::succs(node);
  int outs_count = int32_t(topkv2_outs.size());
  assert(outs_count == 2);

  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_TOPK_V2);
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

  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateTopKV2Options(builder);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_TopKV2Options, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleTranspose *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_TRANSPOSE);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->arg(0)), get_tensor_index(node->arg(1))};
  std::vector<int32_t> outputs_vec{get_tensor_index(node)};

  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateTransposeOptions(builder);

  auto op_offset =
      CreateOperator(builder, op_idx, inputs, outputs,
                     circle::BuiltinOptions::BuiltinOptions_TransposeOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleTransposeConv *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_TRANSPOSE_CONV);

  // Make input, output and options for operator
  std::vector<int32_t> inputs_vec{get_tensor_index(node->inputSizes()),
                                  get_tensor_index(node->filter()),
                                  get_tensor_index(node->outBackprop())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  circle::Padding padding = getOpPadding(node->padding());
  auto options =
      CreateTransposeConvOptions(builder, padding, node->stride()->w(), node->stride()->h());

  // Make TRANSPOSE_CONV operator
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_TransposeConvOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleUnpack *node)
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

  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_UNPACK);
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

  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateUnpackOptions(builder, node->num(), node->axis());
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_UnpackOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleWhile *node)
{
  auto while_outs = loco::succs(node);
  assert(while_outs.size() == node->output_count());

  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_WHILE);
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

  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateWhileOptions(builder, node->cond_branch(), node->body_branch());
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_WhileOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleZerosLike *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_ZEROS_LIKE);
  auto inputs = builder.CreateVector<int32_t>({get_tensor_index(node->input())});
  auto outputs = builder.CreateVector<int32_t>({get_tensor_index(static_cast<loco::Node *>(node))});
  auto options = CreateZerosLikeOptions(builder);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_ZerosLikeOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleBCQFullyConnected *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_BCQ_FULLY_CONNECTED);

  // Make input, output and options for operator
  std::vector<int32_t> inputs_vec{
      get_tensor_index(node->input()), get_tensor_index(node->weights_scales()),
      get_tensor_index(node->weights_binary()), get_tensor_index(node->bias()),
      get_tensor_index(node->weights_clusters())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateBCQFullyConnectedOptions(builder, node->weights_hidden_size(),
                                                to_circle_actfunc(node->fusedActivationFunction()));

  // Make BCQ_FULLY_CONNECTED operator
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_BCQFullyConnectedOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleBCQGather *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_BCQ_GATHER);

  // Make input, output and options for operator
  std::vector<int32_t> inputs_vec{
      get_tensor_index(node->input_scales()), get_tensor_index(node->input_binary()),
      get_tensor_index(node->indices()), get_tensor_index(node->input_clusters())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateBCQGatherOptions(builder, node->input_hidden_size(), node->axis());

  // Make BCQ_GATHER operator
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_BCQGatherOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleInstanceNorm *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_INSTANCE_NORM);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->input()), get_tensor_index(node->gamma()),
                                  get_tensor_index(node->beta())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateInstanceNormOptions(builder, node->epsilon(),
                                           to_circle_actfunc(node->fusedActivationFunction()));
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_InstanceNormOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void exportNode(loco::Node *node, flatbuffers::FlatBufferBuilder &builder, SerializedModelData &md,
                SerializedGraphData &gd)
{
  if (auto circle_node = dynamic_cast<luci::CircleNode *>(node))
  {
    OperationExporter exporter{builder, md, gd};
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
