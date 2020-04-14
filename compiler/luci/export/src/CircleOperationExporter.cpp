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
  OperationExporter(FlatBufferBuilder &fbb, SerializedModelData &ctx) : builder{fbb}, gd{ctx}
  {
    // DO NOTHING
  }

public:
  void visit(luci::CircleAbs *) final;
  void visit(luci::CircleAdd *) final;
  void visit(luci::CircleArgMax *) final;
  void visit(luci::CircleAveragePool2D *) final;
  void visit(luci::CircleConcatenation *) final;
  void visit(luci::CircleConst *) final{/* skip, everything is done in exportOpDefinedTensors */};
  void visit(luci::CircleConv2D *) final;
  void visit(luci::CircleCos *) final;
  void visit(luci::CircleDepthwiseConv2D *) final;
  void visit(luci::CircleDiv *) final;
  void visit(luci::CircleEqual *) final;
  void visit(luci::CircleFullyConnected *) final;
  void visit(luci::CircleLogicalNot *) final;
  void visit(luci::CircleLogicalOr *) final;
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
  void visit(luci::CircleSoftmax *) final;
  void visit(luci::CircleSqrt *) final;
  void visit(luci::CircleSquaredDifference *) final;
  void visit(luci::CircleSub *) final;
  // TODO CircleTanh
  void visit(luci::CircleTranspose *) final;
  void visit(luci::CircleTransposeConv *) final;
  // Circle only
  void visit(luci::CircleInstanceNorm *) final;
  // Virtual
  void visit(luci::CircleInput *) final {}
  void visit(luci::CircleOutput *) final {}

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
  SerializedModelData &gd;
};

template <class CirclePool2D>
void OperationExporter::export_pool_2d(CirclePool2D *node, circle::BuiltinOperator builtin_op)
{
  LUCI_ASSERT(builtin_op == circle::BuiltinOperator_MAX_POOL_2D ||
                  builtin_op == circle::BuiltinOperator_AVERAGE_POOL_2D,
              "Should be MaxPool or AvgPool");
  LUCI_ASSERT(node->padding() != luci::Padding::UNDEFINED, "Padding is not set");

  uint32_t op_idx = gd.registerBuiltinOpcode(builtin_op);
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
  uint32_t op_idx = gd.registerBuiltinOpcode(circle::BuiltinOperator_ABS);
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
  uint32_t op_idx = gd.registerBuiltinOpcode(circle::BuiltinOperator_ADD);
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
  uint32_t op_idx = gd.registerBuiltinOpcode(circle::BuiltinOperator_ARG_MAX);
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

void OperationExporter::visit(luci::CircleConcatenation *node)
{
  uint32_t op_idx = gd.registerBuiltinOpcode(circle::BuiltinOperator_CONCATENATION);
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

void OperationExporter::visit(luci::CircleConv2D *node)
{
  uint32_t op_idx = gd.registerBuiltinOpcode(circle::BuiltinOperator_CONV_2D);

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
  uint32_t op_idx = gd.registerBuiltinOpcode(circle::BuiltinOperator_COS);

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

void OperationExporter::visit(luci::CircleDepthwiseConv2D *node)
{
  uint32_t op_idx = gd.registerBuiltinOpcode(circle::BuiltinOperator_DEPTHWISE_CONV_2D);

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
  uint32_t op_idx = gd.registerBuiltinOpcode(circle::BuiltinOperator_DIV);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->x()), get_tensor_index(node->y())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateDivOptions(builder, to_circle_actfunc(node->fusedActivationFunction()));
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_DivOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleFullyConnected *node)
{
  uint32_t op_idx = gd.registerBuiltinOpcode(circle::BuiltinOperator_FULLY_CONNECTED);

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

void OperationExporter::visit(luci::CircleLogicalNot *node)
{
  uint32_t op_idx = gd.registerBuiltinOpcode(circle::BuiltinOperator_LOGICAL_NOT);

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
  uint32_t op_idx = gd.registerBuiltinOpcode(circle::BuiltinOperator_LOGICAL_OR);

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

void OperationExporter::visit(luci::CircleMaximum *node)
{
  uint32_t op_idx = gd.registerBuiltinOpcode(circle::BuiltinOperator_MAXIMUM);
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
  uint32_t op_idx = gd.registerBuiltinOpcode(circle::BuiltinOperator_MEAN);
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
  uint32_t op_idx = gd.registerBuiltinOpcode(circle::BuiltinOperator_MUL);
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
  uint32_t op_idx = gd.registerBuiltinOpcode(circle::BuiltinOperator_PACK);
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
  uint32_t op_idx = gd.registerBuiltinOpcode(circle::BuiltinOperator_PAD);
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
  uint32_t op_idx = gd.registerBuiltinOpcode(circle::BuiltinOperator_RELU);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->features())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs);
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleRelu6 *node)
{
  uint32_t op_idx = gd.registerBuiltinOpcode(circle::BuiltinOperator_RELU6);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->features())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs);
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleReshape *node)
{
  uint32_t op_idx = gd.registerBuiltinOpcode(circle::BuiltinOperator_RESHAPE);

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
  uint32_t op_idx = gd.registerBuiltinOpcode(circle::BuiltinOperator_RSQRT);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->x())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs);
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleSoftmax *node)
{
  uint32_t op_idx = gd.registerBuiltinOpcode(circle::BuiltinOperator_SOFTMAX);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->logits())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateSoftmaxOptions(builder, node->beta());
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_SoftmaxOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleSqrt *node)
{
  uint32_t op_idx = gd.registerBuiltinOpcode(circle::BuiltinOperator_SQRT);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->x())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs);
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleSquaredDifference *node)
{
  uint32_t op_idx = gd.registerBuiltinOpcode(circle::BuiltinOperator_SQUARED_DIFFERENCE);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->x()), get_tensor_index(node->y())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateSquaredDifferenceOptions(builder);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_SquaredDifferenceOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(luci::CircleSub *node)
{
  uint32_t op_idx = gd.registerBuiltinOpcode(circle::BuiltinOperator_SUB);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->x()), get_tensor_index(node->y())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateSubOptions(builder, to_circle_actfunc(node->fusedActivationFunction()));
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  circle::BuiltinOptions_SubOptions, options.Union());
  gd._operators.push_back(op_offset);
}

// TODO CircleTanh

void OperationExporter::visit(luci::CircleTranspose *node)
{
  uint32_t op_idx = gd.registerBuiltinOpcode(circle::BuiltinOperator_TRANSPOSE);
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
  uint32_t op_idx = gd.registerBuiltinOpcode(circle::BuiltinOperator_TRANSPOSE_CONV);

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

void OperationExporter::visit(luci::CircleInstanceNorm *node)
{
  uint32_t op_idx = gd.registerBuiltinOpcode(circle::BuiltinOperator_INSTANCE_NORM);
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
  uint32_t opcode_idx = gd.registerBuiltinOpcode(circle::BuiltinOperator_EQUAL);
  std::vector<int32_t> inputs{get_tensor_index(node->x()), get_tensor_index(node->y())};
  std::vector<int32_t> outputs{get_tensor_index(node)};

  auto fb_inputs = builder.CreateVector(inputs);
  auto fb_outputs = builder.CreateVector(outputs);

  auto options = CreateEqualOptions(builder);

  auto op_offset = CreateOperator(builder, opcode_idx, fb_inputs, fb_outputs,
                                  circle::BuiltinOptions_EqualOptions, options.Union());

  gd._operators.push_back(op_offset);
}

void exportNode(loco::Node *node, flatbuffers::FlatBufferBuilder &builder,
                SerializedModelData &data)
{
  // TODO Use explicit tagging to prevent possible mistake
  auto isNoOp = [](loco::Node *node) {
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
    OperationExporter exporter{builder, data};
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

void exportNodes(loco::Graph *g, FlatBufferBuilder &builder, SerializedModelData &gd)
{
  for (auto node : loco::postorder_traversal(loco::output_nodes(g)))
  {
    exportNode(node, builder, gd);
  }
}

} // namespace luci
