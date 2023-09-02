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

#include "CircleGen.h"
#include "flatbuffers/flexbuffers.h"

CircleGen::CircleGen() : _subgraph_contexts(1) // Create primary subgraph
{
  // 0th buffer is always the empty buffer for non-const tensors
  addBuffer(nullptr, 0);
}

template <typename T> uint32_t addBuffer(const std::vector<T> &buf_vec)
{
  auto buf = reinterpret_cast<const uint8_t *>(buf_vec.data());
  auto size = buf_vec.size() * sizeof(T);
  return addBuffer(buf, size);
}

uint32_t CircleGen::addBuffer(const uint8_t *buf, size_t size)
{
  uint32_t ind = _buffers.size();
  _buffers.emplace_back(buildBuffer(buf, size));
  return ind;
}

uint32_t CircleGen::addTensor(const TensorParams &params)
{
  uint32_t ind = curSubgCtx().tensors.size();
  curSubgCtx().tensors.emplace_back(buildTensor(params));
  return ind;
}

uint32_t CircleGen::addTensor(const TensorParams &params, float scale, int64_t zero_point)
{
  // TensorType_INT8: scale >= 0, zero_point: [-128, 127]
  // TensorType_UINT8: scale >= 0, zero_point: [0, 255]
  uint32_t ind = curSubgCtx().tensors.size();
  curSubgCtx().tensors.emplace_back(buildTensor(params, scale, zero_point));
  return ind;
}

uint32_t CircleGen::addTensor(const TensorParams &params, std::vector<float> &scale,
                              std::vector<int64_t> &zero_point)
{
  uint32_t ind = curSubgCtx().tensors.size();
  curSubgCtx().tensors.emplace_back(buildTensor(params, scale, zero_point));
  return ind;
}

uint32_t CircleGen::addTensor(const TensorParams &params, const SparsityParams &sp)
{
  uint32_t ind = curSubgCtx().tensors.size();
  curSubgCtx().tensors.emplace_back(buildTensor(params, sp));
  return ind;
}

void CircleGen::setInputsAndOutputs(const std::vector<int> &inputs, const std::vector<int> &outputs)
{
  curSubgCtx().inputs = inputs;
  curSubgCtx().outputs = outputs;
}

uint32_t CircleGen::nextSubgraph()
{
  uint32_t ind = _subgraph_contexts.size();
  _subgraph_contexts.push_back({});
  return ind;
}

CircleBuffer CircleGen::finish()
{
  std::vector<flatbuffers::Offset<circle::SubGraph>> subgraphs;
  for (auto &ctx : _subgraph_contexts)
    subgraphs.push_back(buildSubGraph(ctx));
  auto model =
    circle::CreateModelDirect(_fbb, 3, &_opcodes, &subgraphs, "CircleGen generated", &_buffers);
  _fbb.Finish(model);
  return CircleBuffer{std::move(_fbb)};
}

// ===== Add Operator methods begin =====

uint32_t CircleGen::addOperatorAdd(const OperatorParams &params,
                                   circle::ActivationFunctionType actfn)
{
  auto options = circle::CreateAddOptions(_fbb, actfn).Union();
  return addOperatorWithOptions(params, circle::BuiltinOperator_ADD,
                                circle::BuiltinOptions_AddOptions, options);
}

uint32_t CircleGen::addOperatorAddN(const OperatorParams &params)
{
  auto options = circle::CreateAddNOptions(_fbb).Union();
  return addOperatorWithOptions(params, circle::BuiltinOperator_ADD_N,
                                circle::BuiltinOptions_AddNOptions, options);
}

uint32_t CircleGen::addOperatorArgMax(const OperatorParams &params, circle::TensorType output_type)
{
  auto options = circle::CreateArgMaxOptions(_fbb, output_type).Union();
  return addOperatorWithOptions(params, circle::BuiltinOperator_ARG_MAX,
                                circle::BuiltinOptions_ArgMaxOptions, options);
}

uint32_t CircleGen::addOperatorArgMin(const OperatorParams &params, circle::TensorType output_type)
{
  auto options = circle::CreateArgMaxOptions(_fbb, output_type).Union();
  return addOperatorWithOptions(params, circle::BuiltinOperator_ARG_MIN,
                                circle::BuiltinOptions_ArgMinOptions, options);
}

uint32_t CircleGen::addOperatorAveragePool2D(const OperatorParams &params, circle::Padding padding,
                                             int stride_w, int stride_h, int filter_w, int filter_h,
                                             circle::ActivationFunctionType actfn)
{
  auto options =
    circle::CreatePool2DOptions(_fbb, padding, stride_w, stride_h, filter_w, filter_h, actfn)
      .Union();
  return addOperatorWithOptions(params, circle::BuiltinOperator_AVERAGE_POOL_2D,
                                circle::BuiltinOptions_Pool2DOptions, options);
}

uint32_t CircleGen::addOperatorCast(const OperatorParams &params, circle::TensorType input_type,
                                    circle::TensorType output_type)
{
  auto options = circle::CreateCastOptions(_fbb, input_type, output_type).Union();
  return addOperatorWithOptions(params, circle::BuiltinOperator_CAST,
                                circle::BuiltinOptions_AddOptions, options);
}

uint32_t CircleGen::addOperatorConcatenation(const OperatorParams &params, int axis,
                                             circle::ActivationFunctionType actfn)
{
  auto options = circle::CreateConcatenationOptions(_fbb, axis, actfn).Union();
  return addOperatorWithOptions(params, circle::BuiltinOperator_CONCATENATION,
                                circle::BuiltinOptions_ConcatenationOptions, options);
}

uint32_t CircleGen::addOperatorConv2D(const OperatorParams &params, circle::Padding padding,
                                      int stride_w, int stride_h,
                                      circle::ActivationFunctionType actfn, int dilation_w,
                                      int dilation_h)
{
  auto options =
    circle::CreateConv2DOptions(_fbb, padding, stride_w, stride_h, actfn, dilation_w, dilation_h)
      .Union();
  return addOperatorWithOptions(params, circle::BuiltinOperator_CONV_2D,
                                circle::BuiltinOptions_Conv2DOptions, options);
}

uint32_t CircleGen::addOperatorCos(const OperatorParams &params)
{
  auto options = circle::CreateCosOptions(_fbb).Union();
  return addOperatorWithOptions(params, circle::BuiltinOperator_COS,
                                circle::BuiltinOptions_CosOptions, options);
}

uint32_t CircleGen::addOperatorDepthToSpace(const OperatorParams &params, int32_t block_size)
{
  auto options = circle::CreateDepthToSpaceOptions(_fbb, block_size).Union();
  return addOperatorWithOptions(params, circle::BuiltinOperator_DEPTH_TO_SPACE,
                                circle::BuiltinOptions_DepthToSpaceOptions, options);
}

uint32_t CircleGen::addOperatorDepthwiseConv2D(const OperatorParams &params,
                                               circle::Padding padding, int stride_w, int stride_h,
                                               int depth_multiplier,
                                               circle::ActivationFunctionType actfn, int dilation_w,
                                               int dilation_h)
{
  auto options =
    circle::CreateDepthwiseConv2DOptions(_fbb, padding, stride_w, stride_h, depth_multiplier, actfn,
                                         dilation_w, dilation_h)
      .Union();
  return addOperatorWithOptions(params, circle::BuiltinOperator_DEPTHWISE_CONV_2D,
                                circle::BuiltinOptions_DepthwiseConv2DOptions, options);
}

uint32_t CircleGen::addOperatorDetectionPostProcess(const OperatorParams &params, int num_classes,
                                                    float y_scale, float x_scale, float h_scale,
                                                    float w_scale, float nms_score_threshold,
                                                    float nms_iou_threshold, int max_detections,
                                                    int max_classes_per_detection,
                                                    int detections_per_class)
{
  // flexbuffer custom_option
  auto flex_buffers = std::make_unique<flexbuffers::Builder>();
  size_t map_start = flex_buffers->StartMap();
  flex_buffers->Int("num_classes", num_classes);
  flex_buffers->Float("y_scale", y_scale);
  flex_buffers->Float("x_scale", x_scale);
  flex_buffers->Float("h_scale", h_scale);
  flex_buffers->Float("w_scale", w_scale);
  flex_buffers->Float("nms_iou_threshold", nms_iou_threshold);
  flex_buffers->Float("nms_score_threshold", nms_score_threshold);
  flex_buffers->Int("max_detections", max_detections);
  flex_buffers->Int("max_classes_per_detection", max_classes_per_detection);
  flex_buffers->Int("detections_per_class", detections_per_class);
  flex_buffers->EndMap(map_start);
  flex_buffers->Finish();

  return addCustomOperatorWithOptions(params, "TFLite_Detection_PostProcess",
                                      circle::BuiltinOptions_NONE, 0, &flex_buffers->GetBuffer(),
                                      circle::CustomOptionsFormat::CustomOptionsFormat_FLEXBUFFERS,
                                      nullptr, nullptr);
}

uint32_t CircleGen::addOperatorElu(const OperatorParams &params)
{
  return addOperatorWithOptions(params, circle::BuiltinOperator_ELU, circle::BuiltinOptions_NONE,
                                0);
}

uint32_t CircleGen::addOperatorEqual(const OperatorParams &params)
{
  auto options = circle::CreateEqualOptions(_fbb).Union();
  return addOperatorWithOptions(params, circle::BuiltinOperator_EQUAL,
                                circle::BuiltinOptions_EqualOptions, options);
}

uint32_t CircleGen::addOperatorExpandDims(const OperatorParams &params)
{
  auto options = circle::CreateEqualOptions(_fbb).Union();
  return addOperatorWithOptions(params, circle::BuiltinOperator_EXPAND_DIMS,
                                circle::BuiltinOptions_ExpandDimsOptions, options);
}

uint32_t
CircleGen::addOperatorFullyConnected(const OperatorParams &params,
                                     circle::FullyConnectedOptionsWeightsFormat weights_format)
{
  auto options =
    circle::CreateFullyConnectedOptions(_fbb, circle::ActivationFunctionType_NONE, weights_format)
      .Union();
  return addOperatorWithOptions(params, circle::BuiltinOperator_FULLY_CONNECTED,
                                circle::BuiltinOptions_FullyConnectedOptions, options);
}

uint32_t CircleGen::addOperatorFill(const OperatorParams &params)
{
  auto options = circle::CreateFillOptions(_fbb).Union();
  return addOperatorWithOptions(params, circle::BuiltinOperator_FILL,
                                circle::BuiltinOptions_FillOptions, options);
}

uint32_t CircleGen::addOperatorFloor(const OperatorParams &params)
{
  return addOperatorWithOptions(params, circle::BuiltinOperator_FLOOR, circle::BuiltinOptions_NONE,
                                0);
}

uint32_t CircleGen::addOperatorFloorDiv(const OperatorParams &params)
{
  return addOperatorWithOptions(params, circle::BuiltinOperator_FLOOR_DIV,
                                circle::BuiltinOptions_NONE, 0);
}

uint32_t CircleGen::addOperatorGreater(const OperatorParams &params)
{
  auto options = circle::CreateLessOptions(_fbb).Union();
  return addOperatorWithOptions(params, circle::BuiltinOperator_GREATER,
                                circle::BuiltinOptions_GreaterOptions, options);
}

uint32_t CircleGen::addOperatorGreaterEqual(const OperatorParams &params)
{
  auto options = circle::CreateGreaterOptions(_fbb).Union();
  return addOperatorWithOptions(params, circle::BuiltinOperator_GREATER_EQUAL,
                                circle::BuiltinOptions_GreaterEqualOptions, options);
}

uint32_t CircleGen::addOperatorL2Normalization(const OperatorParams &params)
{
  auto options = circle::CreateL2NormOptions(_fbb).Union();
  return addOperatorWithOptions(params, circle::BuiltinOperator_L2_NORMALIZATION,
                                circle::BuiltinOptions_L2NormOptions, options);
}

uint32_t CircleGen::addOperatorLess(const OperatorParams &params)
{
  auto options = circle::CreateLessOptions(_fbb).Union();
  return addOperatorWithOptions(params, circle::BuiltinOperator_LESS,
                                circle::BuiltinOptions_LessOptions, options);
}

uint32_t CircleGen::addOperatorLessEqual(const OperatorParams &params)
{
  auto options = circle::CreateLessOptions(_fbb).Union();
  return addOperatorWithOptions(params, circle::BuiltinOperator_LESS_EQUAL,
                                circle::BuiltinOptions_LessEqualOptions, options);
}

uint32_t CircleGen::addOperatorLeakyRelu(const OperatorParams &params, float alpha)
{
  auto options = circle::CreateLeakyReluOptions(_fbb, alpha).Union();
  return addOperatorWithOptions(params, circle::BuiltinOperator_LEAKY_RELU,
                                circle::BuiltinOptions_LeakyReluOptions, options);
}

uint32_t CircleGen::addOperatorLogSoftmax(const OperatorParams &params)
{
  auto options = circle::CreateLogSoftmaxOptions(_fbb).Union();
  return addOperatorWithOptions(params, circle::BuiltinOperator_LOG_SOFTMAX,
                                circle::BuiltinOptions_LogSoftmaxOptions, options);
}

uint32_t CircleGen::addOperatorMean(const OperatorParams &params, bool keep_dims)
{
  auto options = circle::CreateReducerOptions(_fbb, keep_dims).Union();
  return addOperatorWithOptions(params, circle::BuiltinOperator_MEAN,
                                circle::BuiltinOptions_ReducerOptions, options);
}

uint32_t CircleGen::addOperatorMul(const OperatorParams &params,
                                   circle::ActivationFunctionType actfn)
{
  auto options = circle::CreateMulOptions(_fbb, actfn).Union();
  return addOperatorWithOptions(params, circle::BuiltinOperator_MUL,
                                circle::BuiltinOptions_MulOptions, options);
}

uint32_t CircleGen::addOperatorNeg(const OperatorParams &params)
{
  auto options = circle::CreatePadOptions(_fbb).Union();
  return addOperatorWithOptions(params, circle::BuiltinOperator_NEG,
                                circle::BuiltinOptions_NegOptions, options);
}

uint32_t CircleGen::addOperatorNotEqual(const OperatorParams &params)
{
  auto options = circle::CreateEqualOptions(_fbb).Union();
  return addOperatorWithOptions(params, circle::BuiltinOperator_NOT_EQUAL,
                                circle::BuiltinOptions_NotEqualOptions, options);
}

uint32_t CircleGen::addOperatorOneHot(const OperatorParams &params, int32_t axis)
{
  auto options = circle::CreateOneHotOptions(_fbb, axis).Union();
  return addOperatorWithOptions(params, circle::BuiltinOperator_ONE_HOT,
                                circle::BuiltinOptions_OneHotOptions, options);
}

uint32_t CircleGen::addOperatorPad(const OperatorParams &params)
{
  auto options = circle::CreatePadOptions(_fbb).Union();
  return addOperatorWithOptions(params, circle::BuiltinOperator_PAD,
                                circle::BuiltinOptions_PadOptions, options);
}

uint32_t CircleGen::addOperatorPadV2(const OperatorParams &params)
{
  auto options = circle::CreatePadOptions(_fbb).Union();
  return addOperatorWithOptions(params, circle::BuiltinOperator_PADV2,
                                circle::BuiltinOptions_PadV2Options, options);
}

uint32_t CircleGen::addOperatorQuantize(const OperatorParams &params)
{
  auto options = circle::CreateQuantizeOptions(_fbb).Union();
  return addOperatorWithOptions(params, circle::BuiltinOperator_QUANTIZE,
                                circle::BuiltinOptions_QuantizeOptions, options);
}

uint32_t CircleGen::addOperatorRank(const OperatorParams &params)
{
  auto options = circle::CreateRankOptions(_fbb).Union();
  return addOperatorWithOptions(params, circle::BuiltinOperator_RANK,
                                circle::BuiltinOptions_RankOptions, options);
}

uint32_t CircleGen::addOperatorReduce(const OperatorParams &params,
                                      circle::BuiltinOperator reduce_op, bool keep_dims)
{
  switch (reduce_op)
  {
    case circle::BuiltinOperator_REDUCE_ANY:
    case circle::BuiltinOperator_REDUCE_MIN:
    case circle::BuiltinOperator_REDUCE_MAX:
    case circle::BuiltinOperator_REDUCE_PROD:
      break;
    default:
      throw std::runtime_error{"Wrong reduce op"};
  }
  auto options = circle::CreateReducerOptions(_fbb, keep_dims).Union();
  return addOperatorWithOptions(params, reduce_op, circle::BuiltinOptions_ReducerOptions, options);
}

uint32_t CircleGen::addOperatorRelu(const OperatorParams &params)
{
  return addOperatorWithOptions(params, circle::BuiltinOperator_RELU, circle::BuiltinOptions_NONE,
                                0);
}

uint32_t CircleGen::addOperatorRelu6(const OperatorParams &params)
{
  return addOperatorWithOptions(params, circle::BuiltinOperator_RELU6, circle::BuiltinOptions_NONE,
                                0);
}

uint32_t CircleGen::addOperatorReshape(const OperatorParams &params, const Shape *new_shape)
{
  auto options = circle::CreateReshapeOptionsDirect(_fbb, new_shape).Union();
  return addOperatorWithOptions(params, circle::BuiltinOperator_RESHAPE,
                                circle::BuiltinOptions_ReshapeOptions, options);
}

uint32_t CircleGen::addOperatorResizeBilinear(const OperatorParams &params, bool align_corners,
                                              bool half_pixel_centers)
{
  auto options =
    circle::CreateResizeBilinearOptions(_fbb, align_corners, half_pixel_centers).Union();
  return addOperatorWithOptions(params, circle::BuiltinOperator_RESIZE_BILINEAR,
                                circle::BuiltinOptions_ResizeBilinearOptions, options);
}

uint32_t CircleGen::addOperatorResizeNearestNeighbor(const OperatorParams &params)
{
  auto options = circle::CreateResizeNearestNeighborOptions(_fbb).Union();
  return addOperatorWithOptions(params, circle::BuiltinOperator_RESIZE_NEAREST_NEIGHBOR,
                                circle::BuiltinOptions_ResizeNearestNeighborOptions, options);
}

uint32_t CircleGen::addOperatorReverseV2(const OperatorParams &params)
{
  auto options = circle::CreateReverseV2Options(_fbb).Union();
  return addOperatorWithOptions(params, circle::BuiltinOperator_REVERSE_V2,
                                circle::BuiltinOptions_ReverseV2Options, options);
}

uint32_t CircleGen::addOperatorShape(const OperatorParams &params, circle::TensorType type)
{
  auto options = circle::CreateShapeOptions(_fbb, type).Union();
  return addOperatorWithOptions(params, circle::BuiltinOperator_SHAPE,
                                circle::BuiltinOptions_RankOptions, options);
}

uint32_t CircleGen::addOperatorSelect(const OperatorParams &params)
{
  auto options = circle::CreateSelectOptions(_fbb).Union();
  return addOperatorWithOptions(params, circle::BuiltinOperator_SELECT,
                                circle::BuiltinOptions_SelectOptions, options);
}

uint32_t CircleGen::addOperatorSelectV2(const OperatorParams &params)
{
  auto options = circle::CreateSelectV2Options(_fbb).Union();
  return addOperatorWithOptions(params, circle::BuiltinOperator_SELECT_V2,
                                circle::BuiltinOptions_SelectV2Options, options);
}

uint32_t CircleGen::addOperatorSlice(const OperatorParams &params)
{
  auto options = circle::CreateSliceOptions(_fbb).Union();
  return addOperatorWithOptions(params, circle::BuiltinOperator_SLICE,
                                circle::BuiltinOptions_SliceOptions, options);
}

uint32_t CircleGen::addOperatorSoftmax(const OperatorParams &params, float beta)
{
  auto options = circle::CreateSoftmaxOptions(_fbb, beta).Union();
  return addOperatorWithOptions(params, circle::BuiltinOperator_SOFTMAX,
                                circle::BuiltinOptions_SoftmaxOptions, options);
}

uint32_t CircleGen::addOperatorSplit(const OperatorParams &params, int32_t num_split)
{
  auto options = circle::CreateSplitOptions(_fbb, num_split).Union();
  return addOperatorWithOptions(params, circle::BuiltinOperator_SPLIT,
                                circle::BuiltinOptions_SplitOptions, options);
}

uint32_t CircleGen::addOperatorStridedSlice(const OperatorParams &params, int32_t begin_mask,
                                            int32_t end_mask, int32_t ellipsis_mask,
                                            int32_t new_axis_mask, int32_t shrink_axis_mask)
{
  auto options = circle::CreateStridedSliceOptions(_fbb, begin_mask, end_mask, ellipsis_mask,
                                                   new_axis_mask, shrink_axis_mask)
                   .Union();
  return addOperatorWithOptions(params, circle::BuiltinOperator_STRIDED_SLICE,
                                circle::BuiltinOptions_StridedSliceOptions, options);
}

uint32_t CircleGen::addOperatorSub(const OperatorParams &params,
                                   circle::ActivationFunctionType actfn)
{
  auto options = circle::CreateSubOptions(_fbb, actfn).Union();
  return addOperatorWithOptions(params, circle::BuiltinOperator_SUB,
                                circle::BuiltinOptions_SubOptions, options);
}

uint32_t CircleGen::addOperatorTile(const OperatorParams &params)
{
  auto options = circle::CreateTileOptions(_fbb).Union();
  return addOperatorWithOptions(params, circle::BuiltinOperator_TILE,
                                circle::BuiltinOptions_TileOptions, options);
}

uint32_t CircleGen::addOperatorWhile(const OperatorParams &params, uint32_t cond_subg,
                                     uint32_t body_subg)
{
  auto options = circle::CreateWhileOptions(_fbb, cond_subg, body_subg).Union();
  return addOperatorWithOptions(params, circle::BuiltinOperator_WHILE,
                                circle::BuiltinOptions_WhileOptions, options);
}

uint32_t CircleGen::addOperatorIf(const OperatorParams &params, uint32_t then_subg,
                                  uint32_t else_subg)
{
  auto options = circle::CreateIfOptions(_fbb, then_subg, else_subg).Union();
  return addOperatorWithOptions(params, circle::BuiltinOperator_IF,
                                circle::BuiltinOptions_IfOptions, options);
}

uint32_t CircleGen::addOperatorInstanceNorm(const OperatorParams &params, float epsilon,
                                            circle::ActivationFunctionType actfn)
{
  auto options = circle::CreateInstanceNormOptions(_fbb, epsilon, actfn).Union();
  return addOperatorWithOptions(params, circle::BuiltinOperator_INSTANCE_NORM,
                                circle::BuiltinOptions_InstanceNormOptions, options);
}

uint32_t CircleGen::addOperatorTranspose(const OperatorParams &params)
{
  auto options = circle::CreateTransposeOptions(_fbb).Union();
  return addOperatorWithOptions(params, circle::BuiltinOperator_TRANSPOSE,
                                circle::BuiltinOptions_TransposeOptions, options);
}

uint32_t CircleGen::addOperatorSqrt(const OperatorParams &params)
{
  return addOperatorWithOptions(params, circle::BuiltinOperator_SQRT, circle::BuiltinOptions_NONE,
                                0);
}

uint32_t CircleGen::addOperatorSquare(const OperatorParams &params)
{
  auto options = circle::CreateSquareOptions(_fbb).Union();
  return addOperatorWithOptions(params, circle::BuiltinOperator_SQUARE,
                                circle::BuiltinOptions_SquareOptions, options);
}

uint32_t CircleGen::addOperatorBatchToSpaceND(const OperatorParams &params)
{
  auto options = circle::CreateBatchToSpaceNDOptions(_fbb).Union();
  return addOperatorWithOptions(params, circle::BuiltinOperator_BATCH_TO_SPACE_ND,
                                circle::BuiltinOptions_BatchToSpaceNDOptions, options);
}

// NOTE Please add addOperator functions ABOVE this lie
//
// %  How to add a new addOperatorXXX fuction
// 0. Copy code from one of the existing addOperatorXXX function
// 1. Change the function signature (need BuiltinOperator params)
// 2. Change enum BuiltinOperator
// 3. Change enum BuiltinOptions
// 4. Change CreateXXXOptions accordingly
//
// If operator don't have option table, remove CreateXXXOptions call,
// call addOperatorWithOptions with options_type = circle::BuiltinOptions_NONE and options = 0

// ===== Add Operator methods end =====

uint32_t CircleGen::addOperatorWithOptions(const OperatorParams &params,
                                           circle::BuiltinOperator opcode,
                                           circle::BuiltinOptions options_type,
                                           flatbuffers::Offset<void> options)
{
  uint32_t opcode_ind = addOperatorCode(opcode);
  auto op = circle::CreateOperatorDirect(_fbb, opcode_ind, &params.inputs, &params.outputs,
                                         options_type, options);

  uint32_t ind = curSubgCtx().operators.size();
  curSubgCtx().operators.emplace_back(op);
  return ind;
}

uint32_t CircleGen::addCustomOperatorWithOptions(
  const OperatorParams &params, std::string custom_code, circle::BuiltinOptions options_type,
  flatbuffers::Offset<void> options, const std::vector<uint8_t> *custom_options,
  circle::CustomOptionsFormat custom_options_format,
  const std::vector<uint8_t> *mutating_variable_inputs, const std::vector<int32_t> *intermediates)

{
  uint32_t opcode_ind = addCustomOperatorCode(custom_code);
  auto op = circle::CreateOperatorDirect(
    _fbb, opcode_ind, &params.inputs, &params.outputs, options_type, options, custom_options,
    custom_options_format, mutating_variable_inputs, intermediates);

  uint32_t ind = curSubgCtx().operators.size();
  curSubgCtx().operators.emplace_back(op);
  return ind;
}

uint32_t CircleGen::addOperatorCode(circle::BuiltinOperator opcode)
{
  // TODO If the same OperatorCode is registered already, just return it
  uint32_t ind = _opcodes.size();
  _opcodes.emplace_back(circle::CreateOperatorCode(_fbb, opcode));
  return ind;
}

uint32_t CircleGen::addCustomOperatorCode(std::string custom_code)
{
  // TODO If the same OperatorCode is registered already, just return it
  uint32_t ind = _opcodes.size();
  _opcodes.emplace_back(
    circle::CreateOperatorCodeDirect(_fbb, circle::BuiltinOperator_CUSTOM, custom_code.c_str()));
  return ind;
}

flatbuffers::Offset<circle::Buffer> CircleGen::buildBuffer(const uint8_t *buf, size_t size)
{
  if (buf == nullptr || size == 0)
    return circle::CreateBuffer(_fbb);
  auto buffer = _fbb.CreateVector(buf, size);
  return circle::CreateBuffer(_fbb, buffer);
}

flatbuffers::Offset<circle::Tensor> CircleGen::buildTensor(const TensorParams &params)
{
  auto shape = _fbb.CreateVector(params.shape);
  auto name = _fbb.CreateString(params.name);
  return circle::CreateTensor(_fbb, shape, params.tensor_type, params.buffer, name,
                              0 /* QuantParam */, false /* is_variable */, 0 /* sparsity */,
                              0 /* shape_signature */);
}

flatbuffers::Offset<circle::Tensor> CircleGen::buildTensor(const TensorParams &params, float scale,
                                                           int64_t zero_point)
{
  auto shape = _fbb.CreateVector(params.shape);
  auto name = _fbb.CreateString(params.name);
  std::vector<float> scale_vector = {scale};
  std::vector<int64_t> zero_point_vector = {zero_point};
  auto quantization = circle::CreateQuantizationParametersDirect(_fbb, nullptr, nullptr,
                                                                 &scale_vector, &zero_point_vector);
  return circle::CreateTensor(_fbb, shape, params.tensor_type, params.buffer, name, quantization,
                              false /* is_variable */, 0 /* sparsity */, 0 /* shape_signature */);
}

flatbuffers::Offset<circle::Tensor> CircleGen::buildTensor(const TensorParams &params,
                                                           std::vector<float> &scales,
                                                           std::vector<int64_t> &zero_points)
{
  auto shape = _fbb.CreateVector(params.shape);
  auto name = _fbb.CreateString(params.name);
  auto quantization =
    circle::CreateQuantizationParametersDirect(_fbb, nullptr, nullptr, &scales, &zero_points);
  return circle::CreateTensor(_fbb, shape, params.tensor_type, params.buffer, name, quantization,
                              false /* is_variable */, 0 /* sparsity */, 0 /* shape_signature */);
}

flatbuffers::Offset<circle::SparsityParameters>
CircleGen::buildSparsityParameters(const SparsityParams &sp)
{
  flatbuffers::Offset<flatbuffers::Vector<int32_t>> traversal_order;
  flatbuffers::Offset<flatbuffers::Vector<int32_t>> block_map;
  flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<circle::DimensionMetadata>>>
    dim_metadata;

  traversal_order = _fbb.CreateVector(sp.traversal_order);
  block_map = _fbb.CreateVector(sp.block_map);

  std::vector<flatbuffers::Offset<circle::DimensionMetadata>> dim_metadata_vec;
  for (auto &it : sp.dim_metadata)
  {
    auto fb_array_segments = circle::CreateUint16VectorDirect(_fbb, &it._array_segments.u16);
    auto fb_array_indices = circle::CreateUint16VectorDirect(_fbb, &it._array_indices.u16);
    auto dim_metadata = circle::CreateDimensionMetadata(
      _fbb, it._format, it._dense_size, it._array_segments_type, fb_array_segments.Union(),
      it._array_indices_type, fb_array_indices.Union());
    dim_metadata_vec.emplace_back(dim_metadata);
  }
  dim_metadata = _fbb.CreateVector(dim_metadata_vec);

  return circle::CreateSparsityParameters(_fbb, traversal_order, block_map, dim_metadata);
}

flatbuffers::Offset<circle::Tensor> CircleGen::buildTensor(const TensorParams &params,
                                                           const SparsityParams &sp)
{
  auto shape = _fbb.CreateVector(params.shape);
  auto name = _fbb.CreateString(params.name);
  auto sparsity = buildSparsityParameters(sp);
  return circle::CreateTensor(_fbb, shape, params.tensor_type, params.buffer, name,
                              0 /* QuantParam */, false /* is_variable */, sparsity,
                              0 /* shape_signature */);
}

flatbuffers::Offset<circle::SubGraph> CircleGen::buildSubGraph(const SubgraphContext &ctx)
{
  return circle::CreateSubGraphDirect(_fbb, &ctx.tensors, &ctx.inputs, &ctx.outputs, &ctx.operators,
                                      nullptr);
}
