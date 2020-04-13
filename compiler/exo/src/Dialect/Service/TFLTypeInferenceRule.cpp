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

#include "TFLTypeInferenceRule.h"

#include "Dialect/IR/TFLDialect.h"
#include "Dialect/IR/TFLNodeVisitor.h"
#include "Dialect/IR/TFLNodes.h"

#include <cassert>

namespace
{

struct TypeInferenceAlgorithm final : public locoex::TFLNodeVisitor<loco::DataType>
{
  loco::DataType visit(const locoex::TFLAdd *node) final { return loco::dtype_get(node->x()); }

  loco::DataType visit(const locoex::TFLAveragePool2D *node) final
  {
    return loco::dtype_get(node->value());
  }

  loco::DataType visit(const locoex::TFLConcatenation *node) final
  {
    // TODO Support when TFLConcatenation has 0 input
    assert(node->numValues() > 0);

    for (uint32_t i = 1; i < node->numValues(); ++i)
      assert(loco::dtype_get(node->values(i - 1)) == loco::dtype_get(node->values(i)));

    return loco::dtype_get(node->values(0));
  }

  loco::DataType visit(const locoex::TFLConst *node) final { return node->dtype(); }

  loco::DataType visit(const locoex::TFLConv2D *node) final
  {
    return loco::dtype_get(node->input());
  }

  loco::DataType visit(const locoex::TFLDepthwiseConv2D *node) final
  {
    return loco::dtype_get(node->input());
  }

  loco::DataType visit(const locoex::TFLDiv *node) final { return loco::dtype_get(node->x()); }

  loco::DataType visit(const locoex::TFLFullyConnected *node) final
  {
    return loco::dtype_get(node->input());
  }

  loco::DataType visit(const locoex::TFLMaximum *node) final { return loco::dtype_get(node->x()); }

  loco::DataType visit(const locoex::TFLMaxPool2D *node) final
  {
    return loco::dtype_get(node->value());
  }

  loco::DataType visit(const locoex::TFLMean *node) final { return loco::dtype_get(node->input()); }

  loco::DataType visit(const locoex::TFLMul *node) final { return loco::dtype_get(node->x()); }

  loco::DataType visit(const locoex::TFLRelu *node) final
  {
    return loco::dtype_get(node->features());
  }

  loco::DataType visit(const locoex::TFLRelu6 *node) final
  {
    return loco::dtype_get(node->features());
  }

  loco::DataType visit(const locoex::TFLReshape *node) final
  {
    return loco::dtype_get(node->tensor());
  }

  loco::DataType visit(const locoex::TFLRsqrt *node) final { return loco::dtype_get(node->x()); }

  // TODO TFLSoftmax

  loco::DataType visit(const locoex::TFLSqrt *node) final { return loco::dtype_get(node->x()); }

  loco::DataType visit(const locoex::TFLSquaredDifference *node) final
  {
    return loco::dtype_get(node->x());
  }

  loco::DataType visit(const locoex::TFLSub *node) final { return loco::dtype_get(node->x()); }

  // TODO TFLTanh

  loco::DataType visit(const locoex::TFLTranspose *node) final
  {
    return loco::dtype_get(node->a());
  }

  loco::DataType visit(const locoex::TFLTransposeConv *node) final
  {
    return loco::dtype_get(node->outBackprop());
  }
};

} // namespace

namespace locoex
{

bool TFLTypeInferenceRule::recognize(const loco::Dialect *d) const
{
  return TFLDialect::get() == d;
}

bool TFLTypeInferenceRule::infer(const loco::Node *node, loco::DataType &dtype) const
{
  assert(node->dialect() == TFLDialect::get());

  TypeInferenceAlgorithm alg;

  dtype = dynamic_cast<const TFLNode *>(node)->accept(&alg);
  assert(dtype != loco::DataType::Unknown);

  return true;
}

} // namespace locoex
