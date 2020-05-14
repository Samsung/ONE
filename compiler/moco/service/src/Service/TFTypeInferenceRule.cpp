/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "moco/Service/TFTypeInferenceRule.h"

#include "moco/IR/TFDialect.h"
#include "moco/IR/TFNodeVisitor.h"
#include "moco/IR/TFNodes.h"

#include "moco/IR/TFNodeImpl.h"

#include <cassert>

namespace
{

using namespace moco;

struct TypeForwardAlgorithm final : public moco::TFNodeVisitor<loco::DataType>
{
  loco::DataType visit(const TFAdd *node) { return dtype_get(node->x()); }
  loco::DataType visit(const TFAvgPool *node) { return dtype_get(node->value()); }
  loco::DataType visit(const TFBiasAdd *node) { return dtype_get(node->value()); }
  loco::DataType visit(const TFConcatV2 *node) { return dtype_get(node->values(0)); }

  loco::DataType visit(const TFConst *node) { return node->dtype(); }

  loco::DataType visit(const TFConv2D *node) { return dtype_get(node->input()); }
  loco::DataType visit(const TFConv2DBackpropInput *node)
  {
    return dtype_get(node->out_backprop());
  }
  loco::DataType visit(const TFDepthwiseConv2dNative *node) { return dtype_get(node->input()); }
  loco::DataType visit(const TFFakeQuantWithMinMaxVars *node) { return dtype_get(node->inputs()); }
  loco::DataType visit(const TFFusedBatchNorm *node) { return dtype_get(node->x()); }
  loco::DataType visit(const TFIdentity *node) { return dtype_get(node->input()); }
  loco::DataType visit(const TFMaximum *node) { return dtype_get(node->x()); }
  loco::DataType visit(const TFMaxPool *node) { return dtype_get(node->input()); }
  loco::DataType visit(const TFMean *node) { return dtype_get(node->input()); }
  loco::DataType visit(const TFMul *node) { return dtype_get(node->x()); }
  loco::DataType visit(const TFPack *node) { return dtype_get(node->values(0)); }
  loco::DataType visit(const TFPad *node) { return dtype_get(node->input()); }

  loco::DataType visit(const TFPlaceholder *node) { return node->dtype(); }

  loco::DataType visit(const TFRealDiv *node) { return dtype_get(node->x()); }
  loco::DataType visit(const TFRelu *node) { return dtype_get(node->features()); }
  loco::DataType visit(const TFRelu6 *node) { return dtype_get(node->features()); }
  loco::DataType visit(const TFReshape *node) { return dtype_get(node->tensor()); }
  loco::DataType visit(const TFRsqrt *node) { return dtype_get(node->x()); }

  loco::DataType visit(const TFShape *node) { return node->dtype(); }

  loco::DataType visit(const TFSoftmax *node) { return dtype_get(node->logits()); }
  loco::DataType visit(const TFSqrt *node) { return dtype_get(node->x()); }
  loco::DataType visit(const TFSquaredDifference *node) { return dtype_get(node->x()); }
  loco::DataType visit(const TFSqueeze *node) { return dtype_get(node->input()); }
  loco::DataType visit(const TFStopGradient *node) { return dtype_get(node->input()); }
  loco::DataType visit(const TFStridedSlice *node) { return dtype_get(node->input()); }
  loco::DataType visit(const TFSub *node) { return dtype_get(node->x()); }
  loco::DataType visit(const TFTanh *node) { return dtype_get(node->x()); }

  // For virtual nodes
  loco::DataType visit(const TFPush *node) { return dtype_get(node->from()); }
};

} // namespace

namespace moco
{

bool TFTypeInferenceRule::recognize(const loco::Dialect *d) const
{
  // This rule recognizes only "TFDialect" dialect!
  return TFDialect::get() == d;
}

bool TFTypeInferenceRule::infer(const loco::Node *node, loco::DataType &dtype) const
{
  assert(node->dialect() == TFDialect::get());

  TypeForwardAlgorithm alg;

// clang-format off
#define TENSORFLOW_NODE(OPCODE,CLASS)                          \
  if (dynamic_cast<const moco::CLASS *>(node))             \
  {                                                            \
    auto tfnode = loco::must_cast<const moco::CLASS *>(node); \
    dtype = tfnode->accept(&alg);                              \
    assert(dtype != loco::DataType::Unknown);                  \
    return true;                                               \
  }
#include "moco/IR/TFNodes.lst"
#undef TENSORFLOW_NODE
  // clang-format on

  return false;
}

} // namespace moco
