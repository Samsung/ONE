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
  void visit(luci::CircleArgMax *) final;
  void visit(luci::CircleAveragePool2D *) final;
  void visit(luci::CircleBatchMatMul *) final;
  void visit(luci::CircleBatchToSpaceND *) final;
  void visit(luci::CircleCast *) final;
  void visit(luci::CircleConcatenation *) final;
  void visit(luci::CircleConst *) final{/* skip, everything is done in exportOpDefinedTensors */};
  void visit(luci::CircleConv2D *) final;
  void visit(luci::CircleCos *) final;
  void visit(luci::CircleCustom *) final;
  void visit(luci::CircleDepthwiseConv2D *) final;
  void visit(luci::CircleDiv *) final;
  void visit(luci::CircleExp *) final;
  void visit(luci::CircleEqual *) final;
  void visit(luci::CircleFullyConnected *) final;
  void visit(luci::CircleGather *) final;
  void visit(luci::CircleIf *) final;
  void visit(luci::CircleLogicalNot *) final;
  void visit(luci::CircleLogicalOr *) final;
  void visit(luci::CircleLogistic *) final;
  void visit(luci::CircleMaximum *) final;
  void visit(luci::CircleMaxPool2D *) final;
  void visit(luci::CircleMean *) final;
  void visit(luci::CircleMul *) final;
  void visit(luci::CirclePack *) final;
  void visit(luci::CirclePad *) final;
  void visit(luci::CircleRelu *) final;
  void visit(luci::CircleRelu6 *) final;
  void visit(luci::CircleReshape *) final;
  void visit(luci::CircleRsqrt *) final;
  void visit(luci::CircleSin *) final;
  void visit(luci::CircleSoftmax *) final;
  void visit(luci::CircleSpaceToBatchND *) final;
  void visit(luci::CircleSqrt *) final;
  void visit(luci::CircleSquaredDifference *) final;
  void visit(luci::CircleStridedSlice *) final;
  void visit(luci::CircleSub *) final;
  void visit(luci::CircleTanh *) final;
  void visit(luci::CircleTile *) final;
  void visit(luci::CircleTranspose *) final;
  void visit(luci::CircleTransposeConv *) final;
  void visit(luci::CircleUnpack *) final;
  void visit(luci::CircleWhile *) final;
  // Circle only
  void visit(luci::CircleInstanceNorm *) final;
  // Virtual
  void visit(luci::CircleInput *) final {}
  void visit(luci::CircleOutput *) final {}
  void visit(luci::CircleOutputDummy *) final {}
  // Virtual for multiple-outputs
  void visit(luci::CircleIfOut *) final {}
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
                  builtin_op == circle::BuiltinOperator_AVERAGE_POOL_2D,
              "Should be MaxPool or AvgPool");
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
  auto options = CreateCastOptions(builder, to_circle_tensortype(node->in_data_type()),
                                   to_circle_tensortype(node->out_data_type()));
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_CastOptions, options.Union());
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
                                     to_circle_actfunc(node->fusedActivationFunction()));

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
  uint32_t op_idx = md.registerCustomOpcode(node->custom_code());
  std::vector<int32_t> inputs_vec;
  for (uint32_t i = 0; i < node->numInputs(); i++)
  {
    inputs_vec.push_back(get_tensor_index(node->inputs(i)));
  }
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
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
                                              to_circle_actfunc(node->fusedActivationFunction()));

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

void OperationExporter::visit(luci::CircleExp *node)
{
  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_EXP);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->x())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateAbsOptions(builder);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_ExpOptions, options.Union());
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
      auto if_out = dynamic_cast<luci::CircleIfOut *>(out);
      assert(if_out != nullptr);
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
  auto unpack_outs = loco::succs(node);
  assert(int32_t(unpack_outs.size()) == node->num());

  uint32_t op_idx = md.registerBuiltinOpcode(circle::BuiltinOperator_UNPACK);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->value())};
  std::vector<int32_t> outputs_vec;

  for (int32_t index = 0; index < node->num(); index++)
  {
    // store in order of index
    bool found = false;
    for (auto out : unpack_outs)
    {
      auto unpack_out = dynamic_cast<luci::CircleUnpackOut *>(out);
      assert(unpack_out != nullptr);
      if (unpack_out->index() == index)
      {
        outputs_vec.push_back(get_tensor_index(unpack_out));
        found = true;
        break;
      }
    }
    if (!found)
    {
      INTERNAL_EXN("Invalid Unpack output");
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
      auto while_out = dynamic_cast<luci::CircleWhileOut *>(out);
      assert(while_out != nullptr);
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

void exportNode(loco::Node *node, flatbuffers::FlatBufferBuilder &builder, SerializedModelData &md,
                SerializedGraphData &gd)
{
  // TODO Use explicit tagging to prevent possible mistake
  auto isNoOp = [](loco::Node *node) {
    if (dynamic_cast<luci::CircleOutputDummy *>(node) != nullptr)
      return true;
    if (dynamic_cast<luci::CircleOutput *>(node) != nullptr)
      return true;
    // If there is only one input and the TensorIndex for the input is same
    // as the TensorIndex of the output then this node is just a dummy node
    if (node->arity() == 1)
    {
      assert(node->arg(0) != nullptr);
      return get_tensor_index(node) == get_tensor_index(node->arg(0));
    }
    return false;
  };

  if (isNoOp(node))
  {
    // Skip if a given node is marked as NoOp (op with no effect) before
    return;
  }

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
