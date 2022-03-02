/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "CircleNodeSummaryBuilders.h"

#include <luci/IR/CircleNode.h>
#include <luci/IR/CircleNodes.h>
#include <loco/IR/Node.h>

#include <string>
#include <vector>

namespace
{

std::string to_str(loco::DataType type)
{
  switch (type)
  {
    case loco::DataType::U8:
      return "UINT8";
    case loco::DataType::U16:
      return "UINT16";
    case loco::DataType::U32:
      return "UINT32";
    case loco::DataType::U64:
      return "UINT64";

    case loco::DataType::S8:
      return "INT8";
    case loco::DataType::S16:
      return "INT16";
    case loco::DataType::S32:
      return "INT32";
    case loco::DataType::S64:
      return "INT64";

    case loco::DataType::FLOAT16:
      return "FLOAT16";
    case loco::DataType::FLOAT32:
      return "FLOAT32";
    case loco::DataType::FLOAT64:
      return "FLOAT64";

    case loco::DataType::BOOL:
      return "BOOL";

    default:
      return "Error";
  }
}

std::string to_str(bool value) { return value ? "true" : "false"; }

std::string to_str(luci::FusedActFunc fused)
{
  switch (fused)
  {
    case luci::FusedActFunc::NONE:
      return "NONE";
    case luci::FusedActFunc::RELU:
      return "RELU";
    case luci::FusedActFunc::RELU_N1_TO_1:
      return "RELU_N1_TO_1";
    case luci::FusedActFunc::RELU6:
      return "RELU6";
    case luci::FusedActFunc::TANH:
      return "TANH";
    case luci::FusedActFunc::SIGN_BIT:
      return "SIGN_BIT";
    default:
      return "Error";
  }
}

std::string to_str(luci::Padding padding)
{
  switch (padding)
  {
    case luci::Padding::SAME:
      return "SAME";
    case luci::Padding::VALID:
      return "VALID";
    default:
      return "Error";
  }
}

std::string to_str(const luci::Stride *stride)
{
  return std::to_string(stride->h()) + "," + std::to_string(stride->w());
}

std::string to_str(const luci::Filter *filter)
{
  return std::to_string(filter->h()) + "," + std::to_string(filter->w());
}

std::string to_str(luci::MirrorPadMode mode)
{
  switch (mode)
  {
    case luci::MirrorPadMode::REFLECT:
      return "REFLECT";
    case luci::MirrorPadMode::SYMMETRIC:
      return "SYMMETRIC";
    default:
      return "Error";
  }
}

} // namespace

namespace luci
{

std::vector<std::string> CircleNodeWithXSummaryBuilder::get_input_names(const luci::CircleNode *)
{
  return {"x"};
}

std::vector<std::string>
CircleNodeWithINPUTSummaryBuilder::get_input_names(const luci::CircleNode *)
{
  return {"input"};
}

std::vector<std::string> CircleNodeWithXYSummaryBuilder::get_input_names(const luci::CircleNode *)
{
  return {"x", "y"};
}

std::vector<std::string>
CircleNodeWithFEATURESSummaryBuilder::get_input_names(const luci::CircleNode *)
{
  return {"features"};
}

} // namespace luci

namespace luci
{

bool CircleAddSummaryBuilder::validate(const luci::CircleNode *node)
{
  auto add = loco::must_cast<const luci::CircleAdd *>(node);
  if (add->fusedActivationFunction() == luci::FusedActFunc::UNDEFINED)
    return false;

  return true;
}

void CircleAddSummaryBuilder::build_attributes(const luci::CircleNode *node, locop::NodeSummary &s)
{
  auto add = loco::must_cast<const luci::CircleAdd *>(node);
  s.args().append("fused_activation_function", to_str(add->fusedActivationFunction()));
}

std::vector<std::string> CircleAddNSummaryBuilder::get_input_names(const luci::CircleNode *node)
{
  return std::vector<std::string>(node->arity(), "inputs");
}

std::vector<std::string> CircleArgMaxSummaryBuilder::get_input_names(const luci::CircleNode *)
{
  return {"input", "dimension"};
}

void CircleArgMaxSummaryBuilder::build_attributes(const luci::CircleNode *node,
                                                  locop::NodeSummary &s)
{
  auto argmax = loco::must_cast<const luci::CircleArgMax *>(node);
  s.args().append("output_type", to_str(argmax->output_type()));
}

std::vector<std::string> CircleArgMinSummaryBuilder::get_input_names(const luci::CircleNode *)
{
  return {"input", "dimension"};
}

void CircleArgMinSummaryBuilder::build_attributes(const luci::CircleNode *node,
                                                  locop::NodeSummary &s)
{
  auto argmin = loco::must_cast<const luci::CircleArgMin *>(node);
  s.args().append("output_type", to_str(argmin->output_type()));
}

bool CircleAveragePool2DSummaryBuilder::validate(const luci::CircleNode *node)
{
  auto avgpool = loco::must_cast<const luci::CircleAveragePool2D *>(node);
  if (avgpool->fusedActivationFunction() == luci::FusedActFunc::UNDEFINED)
    return false;
  if (avgpool->padding() == luci::Padding::UNDEFINED)
    return false;

  return true;
}

std::vector<std::string>
CircleAveragePool2DSummaryBuilder::get_input_names(const luci::CircleNode *)
{
  return {"value"};
}

void CircleAveragePool2DSummaryBuilder::build_attributes(const luci::CircleNode *node,
                                                         locop::NodeSummary &s)
{
  auto avgpool = loco::must_cast<const luci::CircleAveragePool2D *>(node);
  s.args().append("filter(h,w)", to_str(avgpool->filter()));
  s.args().append("stride(h,w)", to_str(avgpool->stride()));
  s.args().append("padding", to_str(avgpool->padding()));
  s.args().append("fused_activation_function", to_str(avgpool->fusedActivationFunction()));
}

void CircleBatchMatMulSummaryBuilder::build_attributes(const luci::CircleNode *node,
                                                       locop::NodeSummary &s)
{
  auto batchmatmul = loco::must_cast<const luci::CircleBatchMatMul *>(node);
  s.args().append("adj_x", to_str(batchmatmul->adj_x()));
  s.args().append("adj_y", to_str(batchmatmul->adj_y()));
}

std::vector<std::string>
CircleBatchToSpaceNDSummaryBuilder::get_input_names(const luci::CircleNode *)
{
  return {"input", "block_shape", "crops"};
}

bool CircleBCQFullyConnectedSummaryBuilder::validate(const luci::CircleNode *node)
{
  auto bcq_fc = loco::must_cast<const luci::CircleBCQFullyConnected *>(node);
  if (bcq_fc->fusedActivationFunction() == luci::FusedActFunc::UNDEFINED)
    return false;

  return true;
}

std::vector<std::string>
CircleBCQFullyConnectedSummaryBuilder::get_input_names(const luci::CircleNode *)
{
  return {"input", "weights_scales", "weights_binary", "bias", "weights_clusters"};
}

void CircleBCQFullyConnectedSummaryBuilder::build_attributes(const luci::CircleNode *node,
                                                             locop::NodeSummary &s)
{
  auto bcq_fc = loco::must_cast<const luci::CircleBCQFullyConnected *>(node);
  s.args().append("fused_activation_function", to_str(bcq_fc->fusedActivationFunction()));
  s.args().append("weights_hidden_size", std::to_string(bcq_fc->weights_hidden_size()));
}

std::vector<std::string> CircleBCQGatherSummaryBuilder::get_input_names(const luci::CircleNode *)
{
  return {"input_scales", "input_binary", "indices", "input_clusters"};
}

void CircleBCQGatherSummaryBuilder::build_attributes(const luci::CircleNode *node,
                                                     locop::NodeSummary &s)
{
  auto bcq_gather = loco::must_cast<const luci::CircleBCQGather *>(node);
  s.args().append("axis", std::to_string(bcq_gather->axis()));
  s.args().append("input_hidden_size", std::to_string(bcq_gather->input_hidden_size()));
}

std::vector<std::string>
CircleBidirectionalSequenceLSTMSummaryBuilder::get_input_names(const luci::CircleNode *)
{
  return {"input",
          "fw_input_to_input_weights",
          "fw_input_to_forget_weights",
          "fw_input_to_cell_weights",
          "fw_input_to_output_weights",
          "fw_recurrent_to_input_weights",
          "fw_recurrent_to_forget_weights",
          "fw_recurrent_to_cell_weights",
          "fw_recurrent_to_output_weights",
          "fw_cell_to_input_weights",
          "fw_cell_to_forget_weights",
          "fw_cell_to_output_weights",
          "fw_input_gate_bias",
          "fw_forget_gate_bias",
          "fw_cell_gate_bias",
          "fw_output_gate_bias",
          "fw_projection_weights",
          "fw_projection_bias",
          "bw_input_to_input_weights",
          "bw_input_to_forget_weights",
          "bw_input_to_cell_weights",
          "bw_input_to_output_weights",
          "bw_recurrent_to_input_weights",
          "bw_recurrent_to_forget_weights",
          "bw_recurrent_to_cell_weights",
          "bw_recurrent_to_output_weights",
          "bw_cell_to_input_weights",
          "bw_cell_to_forget_weights",
          "bw_cell_to_output_weights",
          "bw_input_gate_bias",
          "bw_forget_gate_bias",
          "bw_cell_gate_bias",
          "bw_output_gate_bias",
          "bw_projection_weights",
          "bw_projection_bias",
          "fw_activation_state",
          "fw_cell_state",
          "bw_activation_state",
          "bw_cell_state",
          "auxillary_input",
          "fw_auxillary_input_to_input_weights",
          "fw_auxillary_input_to_forget_weights",
          "fw_auxillary_input_to_cell_weights",
          "fw_auxillary_input_to_output_weights",
          "bw_auxillary_input_to_input_weights",
          "bw_auxillary_input_to_forget_weights",
          "bw_auxillary_input_to_cell_weights",
          "bw_auxillary_input_to_output_weights"};
}

void CircleBidirectionalSequenceLSTMSummaryBuilder::build_attributes(const luci::CircleNode *node,
                                                                     locop::NodeSummary &s)
{
  auto lstm = loco::must_cast<const luci::CircleBidirectionalSequenceLSTM *>(node);
  s.args().append("cell_clip", to_str(lstm->cell_clip()));
  s.args().append("proj_clip", to_str(lstm->proj_clip()));
  s.args().append("merge_outputs", to_str(lstm->merge_outputs()));
  s.args().append("time_major", to_str(lstm->time_major()));
  s.args().append("asymmetric_quantize_inputs", to_str(lstm->asymmetric_quantize_inputs()));
}

std::vector<std::string> CircleCastSummaryBuilder::get_input_names(const luci::CircleNode *)
{
  return {"x"};
}

void CircleCastSummaryBuilder::build_attributes(const luci::CircleNode *node, locop::NodeSummary &s)
{
  auto cast = loco::must_cast<const luci::CircleCast *>(node);
  s.args().append("in_data_type", to_str(cast->in_data_type()));
  s.args().append("out_data_type", to_str(cast->out_data_type()));
}

bool CircleConcatenationSummaryBuilder::validate(const luci::CircleNode *node)
{
  auto concat = loco::must_cast<const luci::CircleConcatenation *>(node);
  if (concat->fusedActivationFunction() == luci::FusedActFunc::UNDEFINED)
    return false;

  return true;
}

std::vector<std::string>
CircleConcatenationSummaryBuilder::get_input_names(const luci::CircleNode *node)
{
  return std::vector<std::string>(node->arity(), "values");
}

void CircleConcatenationSummaryBuilder::build_attributes(const luci::CircleNode *node,
                                                         locop::NodeSummary &s)
{
  auto concat = loco::must_cast<const luci::CircleConcatenation *>(node);
  s.args().append("axis", std::to_string(concat->axis()));
  s.args().append("fused_activation_function", to_str(concat->fusedActivationFunction()));
}

void CircleConstSummaryBuilder::update_status(locop::NodeSummary &s)
{
  s.state(locop::NodeDesc::State::PartiallyKnown);
}

bool CircleConv2DSummaryBuilder::validate(const luci::CircleNode *node)
{
  auto conv2d = loco::must_cast<const luci::CircleConv2D *>(node);
  if (conv2d->fusedActivationFunction() == luci::FusedActFunc::UNDEFINED)
    return false;
  if (conv2d->padding() == luci::Padding::UNDEFINED)
    return false;

  return true;
}

std::vector<std::string> CircleConv2DSummaryBuilder::get_input_names(const luci::CircleNode *)
{
  return {"input", "filter", "bias"};
}

void CircleConv2DSummaryBuilder::build_attributes(const luci::CircleNode *node,
                                                  locop::NodeSummary &s)
{
  auto conv2d = loco::must_cast<const luci::CircleConv2D *>(node);
  s.args().append("stride(h,w)", to_str(conv2d->stride()));
  s.args().append("dilation(h,w)", to_str(conv2d->dilation()));
  s.args().append("padding", to_str(conv2d->padding()));
  s.args().append("fused_activation_function", to_str(conv2d->fusedActivationFunction()));
}

std::vector<std::string> CircleCustomSummaryBuilder::get_input_names(const luci::CircleNode *node)
{
  auto input_names = std::vector<std::string>();
  for (uint32_t i = 0; i < node->arity(); ++i)
    input_names.push_back("input" + std::to_string(i));
  return input_names;
}

void CircleCustomSummaryBuilder::build_attributes(const luci::CircleNode *node,
                                                  locop::NodeSummary &s)
{
  auto custom = loco::must_cast<const luci::CircleCustom *>(node);
  s.args().append("custom_code", custom->custom_code());
}

void CircleDepthToSpaceSummaryBuilder::build_attributes(const luci::CircleNode *node,
                                                        locop::NodeSummary &s)
{
  auto depth_to_space = loco::must_cast<const luci::CircleDepthToSpace *>(node);
  s.args().append("block_size", std::to_string(depth_to_space->block_size()));
}

bool CircleDepthwiseConv2DSummaryBuilder::validate(const luci::CircleNode *node)
{
  auto dw_conv2d = loco::must_cast<const luci::CircleDepthwiseConv2D *>(node);
  if (dw_conv2d->fusedActivationFunction() == luci::FusedActFunc::UNDEFINED)
    return false;
  if (dw_conv2d->padding() == luci::Padding::UNDEFINED)
    return false;

  return true;
}

std::vector<std::string>
CircleDepthwiseConv2DSummaryBuilder::get_input_names(const luci::CircleNode *)
{
  return {"input", "filter", "bias"};
}

void CircleDepthwiseConv2DSummaryBuilder::build_attributes(const luci::CircleNode *node,
                                                           locop::NodeSummary &s)
{
  auto dw_conv2d = loco::must_cast<const luci::CircleDepthwiseConv2D *>(node);
  s.args().append("stride(h,w)", to_str(dw_conv2d->stride()));
  s.args().append("dilation(h,w)", to_str(dw_conv2d->dilation()));
  s.args().append("padding", to_str(dw_conv2d->padding()));
  s.args().append("depthMultiplier", std::to_string(dw_conv2d->depthMultiplier()));
  s.args().append("fused_activation_function", to_str(dw_conv2d->fusedActivationFunction()));
}

std::vector<std::string> CircleExpandDimsSummaryBuilder::get_input_names(const luci::CircleNode *)
{
  return {"input", "axis"};
}

std::vector<std::string> CircleFakeQuantSummaryBuilder::get_input_names(const luci::CircleNode *)
{
  return {"inputs"};
}

void CircleFakeQuantSummaryBuilder::build_attributes(const luci::CircleNode *node,
                                                     locop::NodeSummary &s)
{
  auto fake_quant = loco::must_cast<const luci::CircleFakeQuant *>(node);
  s.args().append("min", std::to_string(fake_quant->min()));
  s.args().append("max", std::to_string(fake_quant->max()));
  s.args().append("num_bits", std::to_string(fake_quant->num_bits()));
  s.args().append("narrow_range", to_str(fake_quant->narrow_range()));
}

std::vector<std::string> CircleFillSummaryBuilder::get_input_names(const luci::CircleNode *)
{
  return {"dims", "value"};
}

bool CircleFullyConnectedSummaryBuilder::validate(const luci::CircleNode *node)
{
  auto fc = loco::must_cast<const luci::CircleFullyConnected *>(node);
  if (fc->fusedActivationFunction() == luci::FusedActFunc::UNDEFINED)
    return false;

  return true;
}

std::vector<std::string>
CircleFullyConnectedSummaryBuilder::get_input_names(const luci::CircleNode *)
{
  return {"input", "weights", "bias"};
}

void CircleFullyConnectedSummaryBuilder::build_attributes(const luci::CircleNode *node,
                                                          locop::NodeSummary &s)
{
  auto fc = loco::must_cast<const luci::CircleFullyConnected *>(node);
  s.args().append("fused_activation_function", to_str(fc->fusedActivationFunction()));
}

std::vector<std::string> CircleGatherSummaryBuilder::get_input_names(const luci::CircleNode *)
{
  return {"params", "indices"};
}

void CircleGatherSummaryBuilder::build_attributes(const luci::CircleNode *node,
                                                  locop::NodeSummary &s)
{
  auto gather = loco::must_cast<const luci::CircleGather *>(node);
  s.args().append("axis", std::to_string(gather->axis()));
}

std::vector<std::string> CircleGatherNdSummaryBuilder::get_input_names(const luci::CircleNode *)
{
  return {"params", "indices"};
}

std::vector<std::string> CircleIfSummaryBuilder::get_input_names(const luci::CircleNode *node)
{
  auto circle_if = loco::must_cast<const luci::CircleIf *>(node);

  auto input_names = std::vector<std::string>();
  input_names.push_back("cond");
  for (uint32_t i = 0; i < circle_if->input_count(); ++i)
    input_names.push_back("input");

  return input_names;
}

void CircleIfSummaryBuilder::build_attributes(const luci::CircleNode *node, locop::NodeSummary &s)
{
  auto circle_if = loco::must_cast<const luci::CircleIf *>(node);

  if (circle_if->then_graph() != nullptr)
    s.args().append("then_graph", circle_if->then_graph()->name());
  else
    s.args().append("then_branch", std::to_string(circle_if->then_branch()));

  if (circle_if->else_graph() != nullptr)
    s.args().append("else_graph", circle_if->else_graph()->name());
  else
    s.args().append("else_branch", std::to_string(circle_if->else_branch()));
}

bool CircleL2NormalizeSummaryBuilder::validate(const luci::CircleNode *node)
{
  auto l2norm = loco::must_cast<const luci::CircleL2Normalize *>(node);
  if (l2norm->fusedActivationFunction() == luci::FusedActFunc::UNDEFINED)
    return false;

  return true;
}

std::vector<std::string> CircleL2NormalizeSummaryBuilder::get_input_names(const luci::CircleNode *)
{
  return {"x"};
}

void CircleL2NormalizeSummaryBuilder::build_attributes(const luci::CircleNode *node,
                                                       locop::NodeSummary &s)
{
  auto l2norm = loco::must_cast<const luci::CircleL2Normalize *>(node);
  s.args().append("fused_activation_function", to_str(l2norm->fusedActivationFunction()));
}

bool CircleL2Pool2DSummaryBuilder::validate(const luci::CircleNode *node)
{
  auto l2pool = loco::must_cast<const luci::CircleL2Pool2D *>(node);
  if (l2pool->fusedActivationFunction() == luci::FusedActFunc::UNDEFINED)
    return false;
  if (l2pool->padding() == luci::Padding::UNDEFINED)
    return false;

  return true;
}

std::vector<std::string> CircleL2Pool2DSummaryBuilder::get_input_names(const luci::CircleNode *)
{
  return {"value"};
}

void CircleL2Pool2DSummaryBuilder::build_attributes(const luci::CircleNode *node,
                                                    locop::NodeSummary &s)
{
  auto l2pool = loco::must_cast<const luci::CircleL2Pool2D *>(node);
  s.args().append("filter(h,w)", to_str(l2pool->filter()));
  s.args().append("stride(h,w)", to_str(l2pool->stride()));
  s.args().append("padding", to_str(l2pool->padding()));
  s.args().append("fused_activation_function", to_str(l2pool->fusedActivationFunction()));
}

void CircleLeakyReluSummaryBuilder::build_attributes(const luci::CircleNode *node,
                                                     locop::NodeSummary &s)
{
  auto leaky_relu = loco::must_cast<const luci::CircleLeakyRelu *>(node);
  s.args().append("alpha", std::to_string(leaky_relu->alpha()));
}

void CircleLocalResponseNormalizationSummaryBuilder::build_attributes(const luci::CircleNode *node,
                                                                      locop::NodeSummary &s)
{
  auto lrn = loco::must_cast<const luci::CircleLocalResponseNormalization *>(node);
  s.args().append("radius", std::to_string(lrn->radius()));
  s.args().append("bias", std::to_string(lrn->bias()));
  s.args().append("alpha", std::to_string(lrn->alpha()));
  s.args().append("beta", std::to_string(lrn->beta()));
}

std::vector<std::string> CircleLogSoftmaxSummaryBuilder::get_input_names(const luci::CircleNode *)
{
  return {"logits"};
}

std::vector<std::string> CircleMatrixDiagSummaryBuilder::get_input_names(const luci::CircleNode *)
{
  return {"diagonal"};
}

std::vector<std::string>
CircleMatrixSetDiagSummaryBuilder::get_input_names(const luci::CircleNode *)
{
  return {"input", "diagonal"};
}

bool CircleMaxPool2DSummaryBuilder::validate(const luci::CircleNode *node)
{
  auto maxpool = loco::must_cast<const luci::CircleMaxPool2D *>(node);
  if (maxpool->fusedActivationFunction() == luci::FusedActFunc::UNDEFINED)
    return false;
  if (maxpool->padding() == luci::Padding::UNDEFINED)
    return false;

  return true;
}

std::vector<std::string> CircleMaxPool2DSummaryBuilder::get_input_names(const luci::CircleNode *)
{
  return {"value"};
}

void CircleMaxPool2DSummaryBuilder::build_attributes(const luci::CircleNode *node,
                                                     locop::NodeSummary &s)
{
  auto maxpool = loco::must_cast<const luci::CircleMaxPool2D *>(node);
  s.args().append("filter(h,w)", to_str(maxpool->filter()));
  s.args().append("stride(h,w)", to_str(maxpool->stride()));
  s.args().append("padding", to_str(maxpool->padding()));
  s.args().append("fused_activation_function", to_str(maxpool->fusedActivationFunction()));
}

bool CircleMirrorPadSummaryBuilder::validate(const luci::CircleNode *node)
{
  auto mirror_pad = loco::must_cast<const luci::CircleMirrorPad *>(node);
  if (mirror_pad->mode() == luci::MirrorPadMode::UNDEFINED)
    return false;

  return true;
}

std::vector<std::string> CircleMirrorPadSummaryBuilder::get_input_names(const luci::CircleNode *)
{
  return {"input", "paddings"};
}

void CircleMirrorPadSummaryBuilder::build_attributes(const luci::CircleNode *node,
                                                     locop::NodeSummary &s)
{
  auto mirror_pad = loco::must_cast<const luci::CircleMirrorPad *>(node);
  s.args().append("mode", to_str(mirror_pad->mode()));
}

bool CircleMulSummaryBuilder::validate(const luci::CircleNode *node)
{
  auto mul = loco::must_cast<const luci::CircleMul *>(node);
  if (mul->fusedActivationFunction() == luci::FusedActFunc::UNDEFINED)
    return false;

  return true;
}

void CircleMulSummaryBuilder::build_attributes(const luci::CircleNode *node, locop::NodeSummary &s)
{
  auto mul = loco::must_cast<const luci::CircleMul *>(node);
  s.args().append("fused_activation_function", to_str(mul->fusedActivationFunction()));
}

std::vector<std::string>
CircleNonMaxSuppressionV4SummaryBuilder::get_input_names(const luci::CircleNode *)
{
  return {"boxes", "scores", "max_output_size", "iou_threshold", "score_threshold"};
}

std::vector<std::string>
CircleNonMaxSuppressionV5SummaryBuilder::get_input_names(const luci::CircleNode *)
{
  return {"boxes",         "scores",          "max_output_size",
          "iou_threshold", "score_threshold", "soft_nms_sigma"};
}

std::vector<std::string> CircleOneHotSummaryBuilder::get_input_names(const luci::CircleNode *)
{
  return {"indices", "depth", "on_value", "off_value"};
}

void CircleOneHotSummaryBuilder::build_attributes(const luci::CircleNode *node,
                                                  locop::NodeSummary &s)
{
  auto onehot = loco::must_cast<const luci::CircleOneHot *>(node);
  s.args().append("axis", std::to_string(onehot->axis()));
}

std::vector<std::string> CircleOutputSummaryBuilder::get_input_names(const luci::CircleNode *)
{
  return {"from"};
}

std::vector<std::string> CirclePackSummaryBuilder::get_input_names(const luci::CircleNode *node)
{
  return std::vector<std::string>(node->arity(), "values");
}

void CirclePackSummaryBuilder::build_attributes(const luci::CircleNode *node, locop::NodeSummary &s)
{
  auto pack = loco::must_cast<const luci::CirclePack *>(node);
  s.args().append("values_count", std::to_string(pack->values_count()));
  s.args().append("axis", std::to_string(pack->axis()));
}

std::vector<std::string> CirclePadSummaryBuilder::get_input_names(const luci::CircleNode *)
{
  return {"input", "paddings"};
}

std::vector<std::string> CirclePadV2SummaryBuilder::get_input_names(const luci::CircleNode *)
{
  return {"input", "paddings", "constant_values"};
}

std::vector<std::string> CirclePReluSummaryBuilder::get_input_names(const luci::CircleNode *)
{
  return {"input", "alpha"};
}

std::vector<std::string> CircleRangeSummaryBuilder::get_input_names(const luci::CircleNode *)
{
  return {"start", "limit", "delta"};
}

std::vector<std::string> CircleReshapeSummaryBuilder::get_input_names(const luci::CircleNode *)
{
  return {"tensor", "shape"};
}

void CircleReshapeSummaryBuilder::update_status(locop::NodeSummary &s)
{
  s.state(locop::NodeDesc::State::PartiallyKnown);
}

std::vector<std::string>
CircleResizeBilinearSummaryBuilder::get_input_names(const luci::CircleNode *)
{
  return {"input", "size"};
}

void CircleResizeBilinearSummaryBuilder::build_attributes(const luci::CircleNode *node,
                                                          locop::NodeSummary &s)
{
  auto resize_bilinear = loco::must_cast<const luci::CircleResizeBilinear *>(node);
  s.args().append("align_corners", to_str(resize_bilinear->align_corners()));
  s.args().append("half_pixel_centers", to_str(resize_bilinear->half_pixel_centers()));
}

std::vector<std::string>
CircleResizeNearestNeighborSummaryBuilder::get_input_names(const luci::CircleNode *)
{
  return {"input", "size"};
}

void CircleResizeNearestNeighborSummaryBuilder::build_attributes(const luci::CircleNode *node,
                                                                 locop::NodeSummary &s)
{
  auto resize_nn = loco::must_cast<const luci::CircleResizeNearestNeighbor *>(node);
  s.args().append("align_corners", to_str(resize_nn->align_corners()));
}

std::vector<std::string>
CircleReverseSequenceSummaryBuilder::get_input_names(const luci::CircleNode *)
{
  return {"input", "seq_lengths"};
}

void CircleReverseSequenceSummaryBuilder::build_attributes(const luci::CircleNode *node,
                                                           locop::NodeSummary &s)
{
  auto reverse_seq = loco::must_cast<const luci::CircleReverseSequence *>(node);
  s.args().append("seq_axis", std::to_string(reverse_seq->seq_axis()));
  s.args().append("batch_axis", std::to_string(reverse_seq->batch_axis()));
}

std::vector<std::string> CircleReverseV2SummaryBuilder::get_input_names(const luci::CircleNode *)
{
  return {"tensor", "axis"};
}

} // namespace luci
