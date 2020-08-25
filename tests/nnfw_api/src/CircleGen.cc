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
  int ind = curSubgCtx().tensors.size();
  curSubgCtx().tensors.emplace_back(buildTensor(params));
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

uint32_t CircleGen::addOperatorCos(const OperatorParams &params)
{
  auto options = circle::CreateCosOptions(_fbb).Union();
  return addOperatorWithOptions(params, circle::BuiltinOperator_COS,
                                circle::BuiltinOptions_CosOptions, options);
}

uint32_t CircleGen::addOperatorL2Normalization(const OperatorParams &params)
{
  auto options = circle::CreateL2NormOptions(_fbb).Union();
  return addOperatorWithOptions(params, circle::BuiltinOperator_L2_NORMALIZATION,
                                circle::BuiltinOptions_L2NormOptions, options);
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

uint32_t CircleGen::addOperatorLess(const OperatorParams &params)
{
  auto options = circle::CreateLessOptions(_fbb).Union();
  return addOperatorWithOptions(params, circle::BuiltinOperator_LESS,
                                circle::BuiltinOptions_LessOptions, options);
}

uint32_t CircleGen::addOperatorWhile(const OperatorParams &params, uint32_t cond_subg,
                                     uint32_t body_subg)
{
  auto options = circle::CreateWhileOptions(_fbb, cond_subg, body_subg).Union();
  return addOperatorWithOptions(params, circle::BuiltinOperator_WHILE,
                                circle::BuiltinOptions_WhileOptions, options);
}

// NOTE Please add addOperator functions ABOVE this lie
//
// %  How to add a new addOperatorXXX fuction
// 0. Copy code from one of the existing addOperatorXXX function
// 1. Change the function signature (need BuiltinOperator params)
// 2. Change enum BuiltinOperator
// 3. Change enum BuiltinOptions
// 4. Change CreateXXXOptions accordingly

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

uint32_t CircleGen::addOperatorCode(circle::BuiltinOperator opcode)
{
  // TODO If the same OperatorCode is registered already, just return it
  uint32_t ind = _opcodes.size();
  _opcodes.emplace_back(circle::CreateOperatorCode(_fbb, opcode));
  return ind;
}

flatbuffers::Offset<circle::Buffer> CircleGen::buildBuffer(const uint8_t *buf, size_t size)
{
  if (buf == nullptr && size == 0)
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

flatbuffers::Offset<circle::SubGraph> CircleGen::buildSubGraph(const SubgraphContext &ctx)
{
  return circle::CreateSubGraphDirect(_fbb, &ctx.tensors, &ctx.inputs, &ctx.outputs, &ctx.operators,
                                      nullptr);
}
