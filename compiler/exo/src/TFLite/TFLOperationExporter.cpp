/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "TFLOperationExporter.h"
#include "TFLExporterUtils.h"
#include "ShapeInference.h"

#include "Dialect/IR/TFLNode.h"
#include "Dialect/IR/TFLNodes.h"
#include "Dialect/IR/TFLNodeVisitor.h"

#include "Check.h"

#include <loco/IR/CanonicalNode.h>
#include <loco/IR/CanonicalNodeVisitor.h>
#include <loco/Service/ShapeInference.h>
#include <locoex/COpCall.h>

#include <oops/InternalExn.h>

#include <flatbuffers/flexbuffers.h>

using namespace flatbuffers;
using namespace tflite;

namespace
{

using namespace exo;
using namespace exo::tflite_detail;

class OperationExporter final : public locoex::TFLNodeMutableVisitor<void>,
                                public loco::CanonicalNodeMutableVisitor<void>
{
public:
  OperationExporter(FlatBufferBuilder &fbb, SerializedModelData &ctx) : builder{fbb}, gd{ctx}
  {
    // DO NOTHING
  }

public:
  // FOR TFLNodes
  void visit(locoex::TFLAdd *) final;
  void visit(locoex::TFLAveragePool2D *) final;
  void visit(locoex::TFLConcatenation *) final;
  void visit(locoex::TFLConst *) final{/* skip, everything is done in exportOpDefinedTensors */};
  void visit(locoex::TFLConv2D *) final;
  void visit(locoex::TFLDepthwiseConv2D *) final;
  void visit(locoex::TFLDiv *) final;
  void visit(locoex::TFLFullyConnected *) final;
  void visit(locoex::TFLMaximum *) final;
  void visit(locoex::TFLMaxPool2D *) final;
  void visit(locoex::TFLMean *) final;
  void visit(locoex::TFLMul *) final;
  void visit(locoex::TFLRelu *) final;
  void visit(locoex::TFLRelu6 *) final;
  // TODO TFLReshape
  void visit(locoex::TFLRsqrt *) final;
  // TODO TFLSoftmax
  void visit(locoex::TFLSqrt *) final;
  void visit(locoex::TFLSquaredDifference *) final;
  void visit(locoex::TFLSub *) final;
  // TODO TFLTanh
  void visit(locoex::TFLTranspose *) final;
  void visit(locoex::TFLTransposeConv *) final;

  // FOR canonical nodes. These will be removed later
  void visit(loco::ReLU *) final;
  void visit(loco::ReLU6 *) final;
  void visit(loco::Tanh *) final;
  void visit(loco::Push *) final
  { /* DO NOTHING */
  }
  void visit(loco::Pull *) final
  { /* DO NOTHING */
  }
  void visit(loco::FeatureEncode *) final;
  void visit(loco::FeatureDecode *) final;
  void visit(loco::FilterEncode *) final;
  void visit(loco::DepthwiseFilterEncode *) final;
  void visit(loco::ConstGen *) final
  { /* skip, everything is done in exportOpDefinedTensors */
  }
  void visit(loco::MaxPool2D *) final;
  void visit(loco::AvgPool2D *) final;
  void visit(loco::Conv2D *) final;
  void visit(loco::TransposedConv2D *) final;
  void visit(loco::DepthwiseConv2D *) final;
  void visit(loco::TensorConcat *) final;
  void visit(loco::TensorReduce *) final;
  void visit(loco::TensorSoftmax *) final;
  void visit(loco::BiasEncode *) final;
  void visit(loco::TensorBiasAdd *) final;
  void visit(loco::FeatureBiasAdd *) final;
  void visit(loco::EltwiseAdd *) final;
  void visit(loco::EltwiseMax *) final;
  void visit(loco::EltwiseMul *) final;
  void visit(loco::EltwiseSub *) final;
  void visit(loco::EltwiseDiv *) final;
  void visit(loco::EltwiseSqrt *) final;
  void visit(loco::FixedReshape *) final;
  void visit(loco::TensorBroadcast *) final;
  void visit(loco::TensorConstantPad *) final;

  void visit(locoex::COpCall *);

private:
  /**
   * @brief Exports TFLMaxPool2D or TFLAveragePool2D
   *
   * @note  TFLPool2D should be one of TFLMaxPool2D or TFLAveragePool2D
   */
  template <class TFLPool2D>
  void export_pool_2d(TFLPool2D *node, tflite::BuiltinOperator builtin_op);

private:
  FlatBufferBuilder &builder;
  SerializedModelData &gd;
};

void OperationExporter::visit(locoex::TFLAdd *node)
{
  uint32_t op_idx = gd.registerBuiltinOpcode(tflite::BuiltinOperator_ADD);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->x()), get_tensor_index(node->y())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateAddOptions(builder, to_tflite_actfunc(node->fusedActivationFunction()));
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  tflite::BuiltinOptions_AddOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(locoex::TFLAveragePool2D *node)
{
  export_pool_2d<locoex::TFLAveragePool2D>(node, tflite::BuiltinOperator_AVERAGE_POOL_2D);
}

void OperationExporter::visit(locoex::TFLConcatenation *node)
{
  uint32_t op_idx = gd.registerBuiltinOpcode(tflite::BuiltinOperator_CONCATENATION);
  std::vector<int32_t> inputs_vec;
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};

  for (uint32_t i = 0; i < node->numValues(); ++i)
    inputs_vec.push_back(get_tensor_index(node->values(i)));

  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateConcatenationOptions(builder, node->axis(),
                                            to_tflite_actfunc(node->fusedActivationFunction()));
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  tflite::BuiltinOptions_ConcatenationOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(locoex::TFLConv2D *node)
{
  uint32_t op_idx = gd.registerBuiltinOpcode(tflite::BuiltinOperator_CONV_2D);

  // Make input, output and options for operator
  std::vector<int32_t> inputs_vec{get_tensor_index(node->input()), get_tensor_index(node->filter()),
                                  get_tensor_index(node->bias())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  tflite::Padding padding = getOpPadding(node->padding());
  auto options = CreateConv2DOptions(builder, padding, node->stride()->w(), node->stride()->h(),
                                     to_tflite_actfunc(node->fusedActivationFunction()));

  // Make CONV_2D operator
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  tflite::BuiltinOptions_Conv2DOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(locoex::TFLDepthwiseConv2D *node)
{
  uint32_t op_idx = gd.registerBuiltinOpcode(tflite::BuiltinOperator_DEPTHWISE_CONV_2D);

  // Make input, output and options for operator
  std::vector<int32_t> inputs_vec{get_tensor_index(node->input()), get_tensor_index(node->filter()),
                                  get_tensor_index(node->bias())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  tflite::Padding padding = getOpPadding(node->padding());
  auto options = CreateDepthwiseConv2DOptions(builder, padding, node->stride()->w(),
                                              node->stride()->h(), node->depthMultiplier(),
                                              to_tflite_actfunc(node->fusedActivationFunction()));

  // Make DEPTHWISE_CONV_2D operator
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  tflite::BuiltinOptions_DepthwiseConv2DOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(locoex::TFLDiv *node)
{
  uint32_t op_idx = gd.registerBuiltinOpcode(tflite::BuiltinOperator_DIV);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->x()), get_tensor_index(node->y())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateDivOptions(builder, to_tflite_actfunc(node->fusedActivationFunction()));
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  tflite::BuiltinOptions_DivOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(locoex::TFLFullyConnected *node)
{
  uint32_t op_idx = gd.registerBuiltinOpcode(tflite::BuiltinOperator_FULLY_CONNECTED);

  // Make input, output and options for operator
  std::vector<int32_t> inputs_vec{get_tensor_index(node->input()),
                                  get_tensor_index(node->weights()),
                                  get_tensor_index(node->bias())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options =
      CreateFullyConnectedOptions(builder, to_tflite_actfunc(node->fusedActivationFunction()));

  // Make FULLY_CONNECTED operator
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  tflite::BuiltinOptions_FullyConnectedOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(locoex::TFLMaximum *node)
{
  uint32_t op_idx = gd.registerBuiltinOpcode(tflite::BuiltinOperator_MAXIMUM);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->x()), get_tensor_index(node->y())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateMaximumMinimumOptions(builder);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  tflite::BuiltinOptions_MaximumMinimumOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(locoex::TFLMaxPool2D *node)
{
  export_pool_2d<locoex::TFLMaxPool2D>(node, tflite::BuiltinOperator_MAX_POOL_2D);
}

void OperationExporter::visit(locoex::TFLMean *node)
{
  uint32_t op_idx = gd.registerBuiltinOpcode(tflite::BuiltinOperator_MEAN);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->input()),
                                  get_tensor_index(node->reduction_indices())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateReducerOptions(builder, node->keep_dims());
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  tflite::BuiltinOptions_ReducerOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(locoex::TFLMul *node)
{
  uint32_t op_idx = gd.registerBuiltinOpcode(tflite::BuiltinOperator_MUL);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->x()), get_tensor_index(node->y())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateMulOptions(builder, to_tflite_actfunc(node->fusedActivationFunction()));
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  tflite::BuiltinOptions_MulOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(locoex::TFLRelu *node)
{
  uint32_t op_idx = gd.registerBuiltinOpcode(tflite::BuiltinOperator_RELU);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->features())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs);
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(locoex::TFLRelu6 *node)
{
  uint32_t op_idx = gd.registerBuiltinOpcode(tflite::BuiltinOperator_RELU6);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->features())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs);
  gd._operators.push_back(op_offset);
}

// TODO TFLReshape

void OperationExporter::visit(locoex::TFLRsqrt *node)
{
  uint32_t op_idx = gd.registerBuiltinOpcode(tflite::BuiltinOperator_RSQRT);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->x())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs);
  gd._operators.push_back(op_offset);
}

// TODO TFLSoftmax

void OperationExporter::visit(locoex::TFLSqrt *node)
{
  uint32_t op_idx = gd.registerBuiltinOpcode(tflite::BuiltinOperator_SQRT);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->x())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs);
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(locoex::TFLSquaredDifference *node)
{
  uint32_t op_idx = gd.registerBuiltinOpcode(tflite::BuiltinOperator_SQUARED_DIFFERENCE);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->x()), get_tensor_index(node->y())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateSquaredDifferenceOptions(builder);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  tflite::BuiltinOptions_SquaredDifferenceOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(locoex::TFLSub *node)
{
  uint32_t op_idx = gd.registerBuiltinOpcode(tflite::BuiltinOperator_SUB);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->x()), get_tensor_index(node->y())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateSubOptions(builder, to_tflite_actfunc(node->fusedActivationFunction()));
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  tflite::BuiltinOptions_SubOptions, options.Union());
  gd._operators.push_back(op_offset);
}

// TODO TFLTanh

void OperationExporter::visit(locoex::TFLTranspose *node)
{
  uint32_t op_idx = gd.registerBuiltinOpcode(tflite::BuiltinOperator_TRANSPOSE);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->arg(0)), get_tensor_index(node->arg(1))};
  std::vector<int32_t> outputs_vec{get_tensor_index(node)};

  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateTransposeOptions(builder);

  auto op_offset =
      CreateOperator(builder, op_idx, inputs, outputs,
                     tflite::BuiltinOptions::BuiltinOptions_TransposeOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(locoex::TFLTransposeConv *node)
{
  uint32_t op_idx = gd.registerBuiltinOpcode(tflite::BuiltinOperator_TRANSPOSE_CONV);

  // Make input, output and options for operator
  std::vector<int32_t> inputs_vec{get_tensor_index(node->inputSizes()),
                                  get_tensor_index(node->filter()),
                                  get_tensor_index(node->outBackprop())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  tflite::Padding padding = getOpPadding(node->padding());
  auto options =
      CreateTransposeConvOptions(builder, padding, node->stride()->w(), node->stride()->h());

  // Make TRANSPOSE_CONV operator
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  tflite::BuiltinOptions_TransposeConvOptions, options.Union());
  gd._operators.push_back(op_offset);
}

template <class TFLPool2D>
void OperationExporter::export_pool_2d(TFLPool2D *node, tflite::BuiltinOperator builtin_op)
{
  EXO_ASSERT(builtin_op == tflite::BuiltinOperator_MAX_POOL_2D ||
                 builtin_op == tflite::BuiltinOperator_AVERAGE_POOL_2D,
             "should be maxpool or avgpool");
  EXO_ASSERT(node->padding() != locoex::Padding::UNDEFINED, "Padding is not set");

  uint32_t op_idx = gd.registerBuiltinOpcode(builtin_op);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->value())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);

  tflite::Padding padding = getOpPadding(node->padding());

  auto options = CreatePool2DOptions(builder, padding, node->stride()->w(), node->stride()->h(),
                                     node->filter()->w(), node->filter()->h(),
                                     to_tflite_actfunc(node->fusedActivationFunction()));
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  tflite::BuiltinOptions_Pool2DOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(loco::ReLU *node)
{
  uint32_t op_idx = gd.registerBuiltinOpcode(tflite::BuiltinOperator_RELU);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->input())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs);
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(loco::ReLU6 *node)
{
  uint32_t op_idx = gd.registerBuiltinOpcode(tflite::BuiltinOperator_RELU6);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->input())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs);
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(loco::Tanh *node)
{
  uint32_t op_idx = gd.registerBuiltinOpcode(tflite::BuiltinOperator_TANH);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->input())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs);
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(loco::MaxPool2D *node)
{
  uint32_t op_idx = gd.registerBuiltinOpcode(tflite::BuiltinOperator_MAX_POOL_2D);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->ifm())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  tflite::Padding padding = getOpPadding(
      node->pad(), node->stride(), ShapeInference::get(node->ifm()), ShapeInference::get(node));
  auto options = CreatePool2DOptions(builder, padding, node->stride()->horizontal(),
                                     node->stride()->vertical(), node->window()->horizontal(),
                                     node->window()->vertical());
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  tflite::BuiltinOptions_Pool2DOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(loco::AvgPool2D *node)
{
  // TFlite only support Valid convention of average pooling
  assert(node->convention() == loco::AvgPool2D::Convention::Valid);

  uint32_t op_idx = gd.registerBuiltinOpcode(tflite::BuiltinOperator_AVERAGE_POOL_2D);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->ifm())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  tflite::Padding padding = getOpPadding(
      node->pad(), node->stride(), ShapeInference::get(node->ifm()), ShapeInference::get(node));
  auto options = CreatePool2DOptions(builder, padding, node->stride()->horizontal(),
                                     node->stride()->vertical(), node->window()->horizontal(),
                                     node->window()->vertical());
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  tflite::BuiltinOptions_Pool2DOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(loco::Conv2D *node)
{
  uint32_t op_idx = gd.registerBuiltinOpcode(tflite::BuiltinOperator_CONV_2D);

  // Third input of CONV_2D of tflite should be bias. We will make (and register to gd) dummy zero
  // bias. Bias would be rank 1, have size of output kernel count, and have all zero values, i.e.
  // zero bias.
  auto *ker = dynamic_cast<loco::FilterEncode *>(node->ker());
  assert(ker);
  int32_t bias_vec_size = ShapeInference::get(ker)._dims[0]; // output kernel count

  auto bias_vec_shape_offset = builder.CreateVector(std::vector<int32_t>{bias_vec_size});
  size_t raw_bias_vec_size = bias_vec_size * sizeof(int32_t);

  std::vector<float> bias_vec_data(bias_vec_size); // initialized as zero vector

  auto bias_vec_offset =
      builder.CreateVector(reinterpret_cast<uint8_t *>(bias_vec_data.data()), raw_bias_vec_size);

  auto bias_buffer_offset = CreateBuffer(builder, bias_vec_offset);

  const auto bias_buffer_id = static_cast<uint32_t>(gd._buffers.size());

  gd._buffers.push_back(bias_buffer_offset);

  auto bias_tensor_id = static_cast<int32_t>(gd._tensors.size());
  auto name_offset = builder.CreateString("t_" + std::to_string(bias_tensor_id));

  auto bias_tensor_offset =
      CreateTensor(builder, bias_vec_shape_offset, TensorType_FLOAT32, bias_buffer_id, name_offset);
  gd._tensors.push_back(bias_tensor_offset);

  // Make input, output and options for operator
  std::vector<int32_t> inputs_vec{get_tensor_index(node->ifm()), get_tensor_index(node->ker()),
                                  bias_tensor_id};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  tflite::Padding padding = getOpPadding(
      node->pad(), node->stride(), ShapeInference::get(node->ifm()), ShapeInference::get(node));
  auto options = CreateConv2DOptions(builder, padding, node->stride()->horizontal(),
                                     node->stride()->vertical());

  // Make CONV_2D operator
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  tflite::BuiltinOptions_Conv2DOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(loco::TransposedConv2D *node)
{
  uint32_t op_idx = gd.registerBuiltinOpcode(tflite::BuiltinOperator_TRANSPOSE_CONV);

  // TRANSPOSE_CONV's first input is output shape array.
  const int32_t outshape_vec_size = 4;
  auto outshape_vec_shape_offset = builder.CreateVector(std::vector<int32_t>{outshape_vec_size});
  size_t raw_outshape_vec_size = outshape_vec_size * sizeof(int32_t);

  std::vector<int32_t> outshape_vec_data(outshape_vec_size);
  {
    // Copy inferred output shape of node
    auto out_feature_shape = loco::shape_get(node).as<loco::FeatureShape>();

    // Feature tensor in TFlite is NHWC
    outshape_vec_data.at(0) = out_feature_shape.count().value();
    outshape_vec_data.at(1) = out_feature_shape.height().value();
    outshape_vec_data.at(2) = out_feature_shape.width().value();
    outshape_vec_data.at(3) = out_feature_shape.depth().value();
  }

  auto outshape_vec_offset = builder.CreateVector(
      reinterpret_cast<uint8_t *>(outshape_vec_data.data()), raw_outshape_vec_size);

  auto outshape_buffer_offset = CreateBuffer(builder, outshape_vec_offset);

  const auto outshape_buffer_id = static_cast<uint32_t>(gd._buffers.size());

  gd._buffers.push_back(outshape_buffer_offset);

  auto outshape_tensor_id = static_cast<int32_t>(gd._tensors.size());
  auto name_offset = builder.CreateString("t_" + std::to_string(outshape_tensor_id));

  auto outshape_tensor_offset = CreateTensor(builder, outshape_vec_shape_offset, TensorType_INT32,
                                             outshape_buffer_id, name_offset);
  gd._tensors.push_back(outshape_tensor_offset);

  // Make input, output and options for operator
  std::vector<int32_t> inputs_vec{outshape_tensor_id, get_tensor_index(node->ker()),
                                  get_tensor_index(node->ifm())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  // NOTE input and output is inversed to use this function
  tflite::Padding padding = getOpPadding(node->pad(), node->stride(), ShapeInference::get(node),
                                         ShapeInference::get(node->ifm()));
  auto options = CreateTransposeConvOptions(builder, padding, node->stride()->horizontal(),
                                            node->stride()->vertical());

  // Make TRANSPOSE_CONV operator
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  tflite::BuiltinOptions_TransposeConvOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(loco::DepthwiseConv2D *node)
{
  uint32_t op_idx = gd.registerBuiltinOpcode(tflite::BuiltinOperator_DEPTHWISE_CONV_2D);

  // Third input of DEPTHWISE_CONV2D of tflite should be bias. We will make (and register to gd)
  // dummy zero bias. Bias would be rank 1, have size of output kernel count, and have all zero
  // values, i.e. zero bias.
  auto *ker = dynamic_cast<loco::DepthwiseFilterEncode *>(node->ker());
  assert(ker);

  int32_t bias_vec_size = ShapeInference::get(ker)._dims[3]; // output_size(C*M)
  auto bias_vec_shape_offset = builder.CreateVector(std::vector<int32_t>{bias_vec_size});

  size_t raw_bias_vec_size = bias_vec_size * sizeof(int32_t);
  std::vector<float> bias_vec_data(bias_vec_size);
  auto bias_vec_offset =
      builder.CreateVector(reinterpret_cast<uint8_t *>(bias_vec_data.data()), raw_bias_vec_size);

  auto bias_buffer_offset = CreateBuffer(builder, bias_vec_offset);

  const auto bias_buffer_id = static_cast<uint32_t>(gd._buffers.size());

  gd._buffers.push_back(bias_buffer_offset);

  auto bias_tensor_id = static_cast<int32_t>(gd._tensors.size());
  auto name_offset = builder.CreateString("t_" + std::to_string(bias_tensor_id));

  auto bias_tensor_offset =
      CreateTensor(builder, bias_vec_shape_offset, TensorType_FLOAT32, bias_buffer_id, name_offset);
  gd._tensors.push_back(bias_tensor_offset);

  std::vector<int32_t> inputs_vec{get_tensor_index(node->ifm()), get_tensor_index(node->ker()),
                                  bias_tensor_id};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  tflite::Padding padding = getOpPadding(
      node->pad(), node->stride(), ShapeInference::get(node->ifm()), ShapeInference::get(node));

  int32_t ifm_channel_size = ShapeInference::get(node->ifm())._dims[3];
  // multiplier = bias_vec_size(output_size)/ifm_channel_size
  auto options =
      CreateDepthwiseConv2DOptions(builder, padding, node->stride()->horizontal(),
                                   node->stride()->vertical(), bias_vec_size / ifm_channel_size);

  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  tflite::BuiltinOptions_DepthwiseConv2DOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(loco::TensorReduce *node)
{
  uint32_t op_idx;

  switch (node->func())
  {
    case loco::ReduceFunc::Mean:
      op_idx = gd.registerBuiltinOpcode(tflite::BuiltinOperator_MEAN);
      break;

    // TODO Support more reduce type operation
    default:
      INTERNAL_EXN_V("Not supported reduce type", oops::to_uint32(node->func()));
  }

  // Create a vector for axes data
  std::vector<int32_t> axes_vec;
  auto rank = ShapeInference::get(node->input())._dims.size();
  for (uint32_t i = 0; i < rank; ++i)
    if (node->axes()->defined(i))
      axes_vec.push_back(i);

  int32_t axes_vec_size = axes_vec.size();
  auto axes_vec_shape_offset = builder.CreateVector(std::vector<int32_t>{axes_vec_size});

  size_t raw_axes_vec_size = axes_vec_size * sizeof(int32_t);
  auto axes_vec_offset =
      builder.CreateVector(reinterpret_cast<uint8_t *>(axes_vec.data()), raw_axes_vec_size);

  auto axes_buffer_offset = CreateBuffer(builder, axes_vec_offset);

  const auto axes_buffer_id = static_cast<uint32_t>(gd._buffers.size());

  gd._buffers.push_back(axes_buffer_offset);

  auto axes_tensor_id = static_cast<int32_t>(gd._tensors.size());
  auto name_offset = builder.CreateString("t_" + std::to_string(axes_tensor_id));

  auto axes_tensor_offset =
      CreateTensor(builder, axes_vec_shape_offset, TensorType_INT32, axes_buffer_id, name_offset);
  gd._tensors.push_back(axes_tensor_offset);

  std::vector<int32_t> inputs_vec{get_tensor_index(node->input()), axes_tensor_id};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateReducerOptions(builder, true); // true is for keep_dims option
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  tflite::BuiltinOptions_ReducerOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(loco::TensorSoftmax *node)
{
  // TODO Support when the input rank of TensorSoftmax is not 2
  assert(ShapeInference::get(node->input())._dims.size() == 2);

  // NOTE TFLite only accepts axis when the value is last dimension
  assert(node->axis() == ShapeInference::get(node->input())._dims.size() - 1);

  uint32_t op_idx = gd.registerBuiltinOpcode(tflite::BuiltinOperator_SOFTMAX);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->input())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateSoftmaxOptions(builder, 1.0f);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  tflite::BuiltinOptions_SoftmaxOptions, options.Union());
  gd._operators.push_back(op_offset);
}

/// @brief Export given node into identity, i.e. CONCATENATION with one input
template <typename NodeT>
void exportIdentity(NodeT *node, FlatBufferBuilder &builder, SerializedModelData &gd)
{
  uint32_t op_idx = gd.registerBuiltinOpcode(tflite::BuiltinOperator_CONCATENATION);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->arg(0))};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateConcatenationOptions(builder); // use dummy 0 axis and NONE activation
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  tflite::BuiltinOptions_ConcatenationOptions, options.Union());

  gd._operators.push_back(op_offset);
}

/// @brief Export loco nodes as TRANSPOSE
void exportAsTranspose(loco::Node *node, FlatBufferBuilder &builder,
                       std::vector<int32_t> &perm_vec_data, SerializedModelData &gd)
{
  uint32_t op_idx = gd.registerBuiltinOpcode(tflite::BuiltinOperator_TRANSPOSE);

  auto options = CreateTransposeOptions(builder);

  // Create constant tensor with perm vector
  constexpr int perm_vec_size = 4;
  assert(perm_vec_data.size() == perm_vec_size);
  auto perm_vec_shape_offset = builder.CreateVector(std::vector<int32_t>{perm_vec_size});
  constexpr size_t raw_perm_vec_size = perm_vec_size * sizeof(int32_t);

  auto perm_vec_offset =
      builder.CreateVector(reinterpret_cast<uint8_t *>(perm_vec_data.data()), raw_perm_vec_size);

  auto perm_buffer_offset = CreateBuffer(builder, perm_vec_offset);

  const auto perm_buffer_id = static_cast<uint32_t>(gd._buffers.size());

  gd._buffers.push_back(perm_buffer_offset);

  auto perm_tensor_id = static_cast<int32_t>(gd._tensors.size());
  auto name_offset = builder.CreateString("t_" + std::to_string(perm_tensor_id));

  auto perm_tensor_offset =
      CreateTensor(builder, perm_vec_shape_offset, TensorType_INT32, perm_buffer_id, name_offset);
  gd._tensors.push_back(perm_tensor_offset);

  // Create permutation node

  std::vector<int32_t> inputs_vec{get_tensor_index(node->arg(0)), perm_tensor_id};
  std::vector<int32_t> outputs_vec{get_tensor_index(node)};

  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);

  constexpr auto options_type = tflite::BuiltinOptions::BuiltinOptions_TransposeOptions;

  auto transpose_offset =
      CreateOperator(builder, op_idx, inputs, outputs, options_type, options.Union());
  gd._operators.push_back(transpose_offset);
}

void OperationExporter::visit(loco::FeatureEncode *node)
{
  auto encoder = dynamic_cast<loco::PermutingEncoder<loco::Domain::Feature> *>(node->encoder());
  auto perm = encoder->perm();

  if (isNHWC(perm))
  {
    // Note that tflite represents feature as NHWC
    exportIdentity(node, builder, gd);
  }
  else
  {
    std::vector<int32_t> perm_vec_data(4);
    perm_vec_data[0] = perm->axis(loco::FeatureAxis::Count);
    perm_vec_data[1] = perm->axis(loco::FeatureAxis::Height);
    perm_vec_data[2] = perm->axis(loco::FeatureAxis::Width);
    perm_vec_data[3] = perm->axis(loco::FeatureAxis::Depth);

    exportAsTranspose(node, builder, perm_vec_data, gd);
  }
}

void OperationExporter::visit(loco::FeatureDecode *node)
{
  auto decoder = dynamic_cast<loco::PermutingDecoder<loco::Domain::Feature> *>(node->decoder());
  auto perm = decoder->perm();

  if (isNHWC(perm))
  {
    // Note that tflite represents feature as NHWC
    exportIdentity(node, builder, gd);
  }
  else
  {
    std::vector<int32_t> perm_vec_data(4);
    perm_vec_data[perm->axis(loco::FeatureAxis::Count)] = 0;
    perm_vec_data[perm->axis(loco::FeatureAxis::Height)] = 1;
    perm_vec_data[perm->axis(loco::FeatureAxis::Width)] = 2;
    perm_vec_data[perm->axis(loco::FeatureAxis::Depth)] = 3;

    exportAsTranspose(node, builder, perm_vec_data, gd);
  }
}

void OperationExporter::visit(loco::FilterEncode *node)
{
  auto encoder = dynamic_cast<loco::PermutingEncoder<loco::Domain::Filter> *>(node->encoder());
  auto perm = encoder->perm();

  if (isNHWC(perm))
  {
    // Note that tflite represents filter as NHWC
    exportIdentity(node, builder, gd);
  }
  else
  {
    std::vector<int32_t> perm_vec_data(4);
    // NOTE In tflite, all tensors means NHWC, so 0 = N, 1 = H, 2 = W, 3 = C
    perm_vec_data[0] = perm->axis(loco::FilterAxis::Count);
    perm_vec_data[1] = perm->axis(loco::FilterAxis::Height);
    perm_vec_data[2] = perm->axis(loco::FilterAxis::Width);
    perm_vec_data[3] = perm->axis(loco::FilterAxis::Depth);

    exportAsTranspose(node, builder, perm_vec_data, gd);
  }
}

void exportAsReshape(loco::Node *node, FlatBufferBuilder &builder,
                     std::vector<int32_t> &new_shape_vec, SerializedModelData &gd)
{
  // NOTE TFLite has two ways to get new shape paramter,
  //      one is by attribute 'new_shape' and the other is by input 'shape'.
  //      Therefore TFLite interpreter calculates Reshape operation correctly
  //      if one of them is valid.
  //      However, since NN runtime usually get new shape parameter by input 'shape',
  //      passing new shape only by attribute can cause some problems.
  //      Of course, the opposite situation can be occurred in the future.
  //      To prevent those problems, we pass new shape parameter not only by attribute
  //      but also by input.

  auto input_shape_shape_vec_offset =
      builder.CreateVector(std::vector<int32_t>{(int32_t)new_shape_vec.size()});

  size_t input_shape_vec_size = new_shape_vec.size() * sizeof(int32_t);
  auto input_shape_input_vec_offset =
      builder.CreateVector(reinterpret_cast<uint8_t *>(new_shape_vec.data()), input_shape_vec_size);
  auto input_shape_buffer_offset = CreateBuffer(builder, input_shape_input_vec_offset);

  const auto input_shape_buffer_id = static_cast<uint32_t>(gd._buffers.size());
  gd._buffers.push_back(input_shape_buffer_offset);

  auto input_shape_tensor_id = static_cast<int32_t>(gd._tensors.size());
  auto name_offset = builder.CreateString("t_" + std::to_string(input_shape_tensor_id));
  auto input_shape_tensor_offset = CreateTensor(
      builder, input_shape_shape_vec_offset, TensorType_INT32, input_shape_buffer_id, name_offset);
  gd._tensors.push_back(input_shape_tensor_offset);

  uint32_t op_idx = gd.registerBuiltinOpcode(tflite::BuiltinOperator_RESHAPE);

  std::vector<int32_t> inputs_vec{get_tensor_index(node->arg(0)), input_shape_tensor_id};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);

  auto new_shape_vec_offset = builder.CreateVector(new_shape_vec);
  auto options = CreateReshapeOptions(builder, new_shape_vec_offset);

  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  tflite::BuiltinOptions_ReshapeOptions, options.Union());

  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(loco::DepthwiseFilterEncode *node)
{
  auto ker = node->input(); // [H, W, C, M]

  // tflite represents filter as [1, H, W, C*M] where M is multiplier.
  std::vector<int32_t> new_shape_vec(4);
  new_shape_vec[0] = 1;
  new_shape_vec[1] = ShapeInference::get(ker)._dims[0];
  new_shape_vec[2] = ShapeInference::get(ker)._dims[1];
  new_shape_vec[3] = ShapeInference::get(ker)._dims[2] * ShapeInference::get(ker)._dims[3];

  exportAsReshape(node, builder, new_shape_vec, gd);
}

void OperationExporter::visit(loco::BiasAdd<loco::Domain::Tensor> *node)
{
  uint32_t op_idx = gd.registerBuiltinOpcode(tflite::BuiltinOperator_ADD);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->value()), get_tensor_index(node->bias())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateAddOptions(builder); // dummy option
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  tflite::BuiltinOptions_AddOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(loco::FeatureBiasAdd *node)
{
  uint32_t op_idx = gd.registerBuiltinOpcode(tflite::BuiltinOperator_ADD);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->value()), get_tensor_index(node->bias())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateAddOptions(builder); // dummy option
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  tflite::BuiltinOptions_AddOptions, options.Union());
  gd._operators.push_back(op_offset);
}

/// @brief Export CONCATENATION of **TWO** tensors only
void OperationExporter::visit(loco::TensorConcat *node)
{
  uint32_t op_idx = gd.registerBuiltinOpcode(tflite::BuiltinOperator_CONCATENATION);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->lhs()), get_tensor_index(node->rhs())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateConcatenationOptions(builder, node->axis());
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  tflite::BuiltinOptions_ConcatenationOptions, options.Union());

  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(loco::BiasEncode *encode) { exportIdentity(encode, builder, gd); }

void OperationExporter::visit(loco::EltwiseAdd *node)
{
  uint32_t op_idx = gd.registerBuiltinOpcode(tflite::BuiltinOperator_ADD);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->lhs()), get_tensor_index(node->rhs())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateAddOptions(builder); // dummy option
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  tflite::BuiltinOptions_AddOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(loco::EltwiseMax *node)
{
  uint32_t op_idx = gd.registerBuiltinOpcode(tflite::BuiltinOperator_MAXIMUM);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->lhs()), get_tensor_index(node->rhs())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateMaximumMinimumOptions(builder); // dummy option
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  tflite::BuiltinOptions_MaximumMinimumOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(loco::EltwiseMul *node)
{
  uint32_t op_idx = gd.registerBuiltinOpcode(tflite::BuiltinOperator_MUL);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->lhs()), get_tensor_index(node->rhs())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateMulOptions(builder); // dummy option
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  tflite::BuiltinOptions_MulOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(loco::EltwiseSub *node)
{
  uint32_t op_idx = gd.registerBuiltinOpcode(tflite::BuiltinOperator_SUB);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->lhs()), get_tensor_index(node->rhs())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateSubOptions(builder); // dummy option
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  tflite::BuiltinOptions_SubOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(loco::EltwiseDiv *node)
{
  uint32_t op_idx = gd.registerBuiltinOpcode(tflite::BuiltinOperator_DIV);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->lhs()), get_tensor_index(node->rhs())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto options = CreateDivOptions(builder); // dummy option
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  tflite::BuiltinOptions_DivOptions, options.Union());
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(loco::EltwiseSqrt *node)
{
  uint32_t op_idx = gd.registerBuiltinOpcode(tflite::BuiltinOperator_SQRT);
  std::vector<int32_t> inputs_vec{get_tensor_index(node->input())};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs);
  gd._operators.push_back(op_offset);
}

void OperationExporter::visit(loco::FixedReshape *node)
{
  std::vector<int32_t> new_shape_vec;
  for (uint32_t axis = 0; axis < node->rank(); ++axis)
  {
    assert(node->dim(axis).known());
    new_shape_vec.push_back(node->dim(axis).value());
  }

  exportAsReshape(node, builder, new_shape_vec, gd);
}

void OperationExporter::visit(loco::TensorBroadcast *)
{
  INTERNAL_EXN("TensorBroadcast should not exist in the graph");
}

void OperationExporter::visit(loco::TensorConstantPad *node)
{
  uint32_t op_idx = gd.registerBuiltinOpcode(tflite::BuiltinOperator_PAD);

  // make padding attribute an input
  auto padding = node->padding();
  // get padding vector size
  int32_t padding_vec_size = padding->rank();
  // get byte size of vector
  size_t padding_vec_byte_size = padding_vec_size * sizeof(int32_t) * 2; // [rank, 2]
  // create vector for data
  std::vector<int32_t> padding_vec_data(padding_vec_size * 2);
  // set data
  for (int32_t i = 0; i < padding_vec_size; i++)
  {
    padding_vec_data.at(i * 2) = padding->front(i);
    padding_vec_data.at(i * 2 + 1) = padding->back(i);
  }
  // create FlatBuffer vector
  auto padding_vec_ptr = builder.CreateVector(reinterpret_cast<uint8_t *>(padding_vec_data.data()),
                                              padding_vec_byte_size);

  // create buffer
  auto padding_buffer_ptr = CreateBuffer(builder, padding_vec_ptr);
  // get buffer id
  const auto padding_buffer_id = static_cast<uint32_t>(gd._buffers.size());

  gd._buffers.push_back(padding_buffer_ptr);

  // create padding shape vector
  auto padding_shape_vec_ptr = builder.CreateVector(std::vector<int32_t>{padding_vec_size, 2});
  // create tensor
  auto padding_tensor_ptr =
      CreateTensor(builder, padding_shape_vec_ptr, TensorType_INT32, padding_buffer_id);
  // get tensor id
  const auto padding_tensor_id = static_cast<int32_t>(gd._tensors.size());

  gd._tensors.push_back(padding_tensor_ptr);

  std::vector<int32_t> inputs_vec{get_tensor_index(node->input()), padding_tensor_id};
  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(node))};
  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs);
  gd._operators.push_back(op_offset);
}

inline flatbuffers::Offset<flatbuffers::Vector<uint8_t>>
CreateCOpCallOptions(flatbuffers::FlatBufferBuilder &fbb, locoex::COpCall *copCall)
{
  // read attrs in FlexBuffer format and pass them to FlatBuffer builder
  flexbuffers::Builder flexbuf;
  {
    size_t map_start = flexbuf.StartMap();

    // Note: among attrs of COpCall, 'op' and 'name' won't be included into tflite file
    auto names = copCall->attr_names();
    for (auto name : names)
    {
      if (auto int_val = copCall->attr<locoex::COpAttrType::Int>(name))
        flexbuf.Int(name.c_str(), int_val->val());
      else if (auto float_val = copCall->attr<locoex::COpAttrType::Float>(name))
        flexbuf.Float(name.c_str(), float_val->val());
      else
        // TODO Support more attribute types
        INTERNAL_EXN("Not supported type while writing flexbuffer");
    }

    flexbuf.EndMap(map_start);
    flexbuf.Finish();
  }

  auto offset = fbb.CreateVector(flexbuf.GetBuffer());

  return offset;
}

void OperationExporter::visit(locoex::COpCall *call)
{
  // Registering this custom op name into tflite Operator Codes table
  uint32_t op_idx = gd.registerCustomOpcode(call->op());

  std::vector<int32_t> inputs_vec;
  {
    inputs_vec.resize(call->arity());
    for (uint32_t i = 0; i < call->arity(); i++)
      inputs_vec[i] = get_tensor_index(call->arg(i));
  }

  std::vector<int32_t> outputs_vec{get_tensor_index(static_cast<loco::Node *>(call))};

  auto inputs = builder.CreateVector(inputs_vec);
  auto outputs = builder.CreateVector(outputs_vec);

  auto custom_options = CreateCOpCallOptions(builder, call);
  auto op_offset = CreateOperator(builder, op_idx, inputs, outputs,
                                  tflite::BuiltinOptions_NONE, // builtin_options_type
                                  0,                           // built-in option
                                  custom_options,              // custom options
                                  tflite::CustomOptionsFormat_FLEXBUFFERS);

  gd._operators.push_back(op_offset);
}

void exportNode(loco::Node *node, flatbuffers::FlatBufferBuilder &builder,
                SerializedModelData &data)
{
  // TODO Use explicit tagging to prevent possible mistake
  auto isNoOp = [](loco::Node *node) {
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

  if (auto canonical_node = dynamic_cast<loco::CanonicalNode *>(node))
  { // TODO Consider removing this later
    OperationExporter exporter{builder, data};
    canonical_node->accept(&exporter);
  }
  else if (auto tfl_node = dynamic_cast<locoex::TFLNode *>(node))
  {
    OperationExporter exporter{builder, data};
    tfl_node->accept(&exporter);
  }
  else if (dynamic_cast<locoex::COpNode *>(node))
  {
    OperationExporter exporter{builder, data};
    exporter.visit(dynamic_cast<locoex::COpCall *>(node));
  }
  else
  {
    assert(false && "unsupported node found");
  }
}

} // namespace

namespace exo
{
namespace tflite_detail
{

void exportNodes(loco::Graph *g, FlatBufferBuilder &builder, SerializedModelData &gd)
{
  for (auto node : loco::postorder_traversal(loco::output_nodes(g)))
  {
    exportNode(node, builder, gd);
  }
}

} // namespace tflite_detail
} // namespace exo
