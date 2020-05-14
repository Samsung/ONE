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

#include "loco/Service/TypeInference.h"

#include "loco/IR/Algorithm.h"

#include <cassert>

#include <stdex/Memory.h>

namespace
{

struct DataTypeAnnotation : public loco::NodeAnnotation
{
public:
  DataTypeAnnotation(const loco::DataType &dtype) : _dtype{dtype}
  {
    // DO NOTHING
  }

public:
  const loco::DataType &dtype(void) const { return _dtype; }

private:
  loco::DataType _dtype;
};

bool inputs_dtype_ready(loco::Node *node)
{
  assert(node != nullptr);

  for (uint32_t arity = 0; arity < node->arity(); ++arity)
  {
    if (!loco::TypeInference::known(node->arg(arity)))
    {
      return false;
    }
  }
  return true;
}

} // namespace

namespace loco
{

bool TypeInferenceSession::to(Graph *g) const
{
  bool changed = false;

  for (auto node : postorder_traversal(output_nodes(g)))
  {
    if (_rule->recognize(node->dialect()))
    {
      DataType dtype = DataType::Unknown;

      if (!dtype_known(node) && inputs_dtype_ready(node))
      {
        if (_rule->infer(node, dtype))
        {
          node->annot(stdex::make_unique<DataTypeAnnotation>(dtype));
          changed = true;
        }
      }
    }
  }

  return changed;
}

bool TypeInference::known(const Node *node) { return node->annot<DataTypeAnnotation>() != nullptr; }

DataType TypeInference::get(const Node *node)
{
  assert(known(node));
  return node->annot<DataTypeAnnotation>()->dtype();
}

void TypeInference::erase(Node *node) { return node->annot<DataTypeAnnotation>(nullptr); }

} // namespace loco

//
// Canonical (Data) Type Inference Rule
//
#include <loco/IR/CanonicalDialect.h>
#include <loco/IR/CanonicalNode.h>
#include <loco/IR/CanonicalNodeVisitor.h>

namespace
{

/**
 * There are two possible maintenance policies.
 * - Introduce a new canonical node first, and then extend this algorithm later
 * - Introduce a new canonical node and extend this algorithm at the same time
 *
 * The current implementation assumes the former one (for historical reason).
 *
 * TODO Evaluate the impact of the latter one
 */
struct CanonicalTypeForwardAlgorithm final : public loco::CanonicalNodeVisitor<loco::DataType>
{
  loco::DataType visit(const loco::AvgPool2D *node) { return loco::dtype_get(node->ifm()); }
  loco::DataType visit(const loco::BiasDecode *node) { return loco::dtype_get(node->input()); }
  loco::DataType visit(const loco::BiasEncode *node) { return loco::dtype_get(node->input()); }
  loco::DataType visit(const loco::ConstGen *node) { return node->dtype(); }
  loco::DataType visit(const loco::Conv2D *node) { return loco::dtype_get(node->ifm()); }
  loco::DataType visit(const loco::DepthwiseConv2D *node) { return loco::dtype_get(node->ifm()); }
  loco::DataType visit(const loco::DepthwiseFilterEncode *node)
  {
    return loco::dtype_get(node->input());
  }
  loco::DataType visit(const loco::DepthwiseFilterDecode *node)
  {
    return loco::dtype_get(node->input());
  }
  loco::DataType visit(const loco::EltwiseAdd *node) { return loco::dtype_get(node->lhs()); }
  loco::DataType visit(const loco::EltwiseDiv *node) { return loco::dtype_get(node->lhs()); }
  loco::DataType visit(const loco::EltwiseMax *node) { return loco::dtype_get(node->lhs()); }
  loco::DataType visit(const loco::EltwiseMul *node) { return loco::dtype_get(node->lhs()); }
  loco::DataType visit(const loco::EltwiseSqrt *node) { return loco::dtype_get(node->input()); }
  loco::DataType visit(const loco::EltwiseSub *node) { return loco::dtype_get(node->lhs()); }
  loco::DataType visit(const loco::Forward *node) { return loco::dtype_get(node->input()); }
  loco::DataType visit(const loco::FeatureBiasAdd *node) { return loco::dtype_get(node->value()); }
  loco::DataType visit(const loco::FeatureDecode *node) { return loco::dtype_get(node->input()); }
  loco::DataType visit(const loco::FeatureEncode *node) { return loco::dtype_get(node->input()); }
  loco::DataType visit(const loco::FilterDecode *node) { return loco::dtype_get(node->input()); }
  loco::DataType visit(const loco::FilterEncode *node) { return loco::dtype_get(node->input()); }
  loco::DataType visit(const loco::FixedReshape *node) { return loco::dtype_get(node->input()); }
  loco::DataType visit(const loco::MatrixDecode *node) { return loco::dtype_get(node->input()); }
  loco::DataType visit(const loco::MatrixEncode *node) { return loco::dtype_get(node->input()); }
  loco::DataType visit(const loco::MatMul *node) { return loco::dtype_get(node->lhs()); }
  loco::DataType visit(const loco::MaxPool2D *node) { return loco::dtype_get(node->ifm()); }
  loco::DataType visit(const loco::Push *node) { return loco::dtype_get(node->from()); }
  loco::DataType visit(const loco::Pull *node) { return node->dtype(); }
  loco::DataType visit(const loco::ReLU *node) { return loco::dtype_get(node->input()); }
  loco::DataType visit(const loco::ReLU6 *node) { return loco::dtype_get(node->input()); }
  loco::DataType visit(const loco::Tanh *node) { return loco::dtype_get(node->input()); }
  loco::DataType visit(const loco::TensorConcat *node) { return loco::dtype_get(node->lhs()); }
  loco::DataType visit(const loco::TensorConstantPad *node)
  {
    return loco::dtype_get(node->input());
  }
  loco::DataType visit(const loco::TensorBiasAdd *node) { return loco::dtype_get(node->value()); }
  loco::DataType visit(const loco::TensorBroadcast *node) { return loco::dtype_get(node->input()); }
  loco::DataType visit(const loco::TensorReduce *node) { return loco::dtype_get(node->input()); }
  loco::DataType visit(const loco::TensorSoftmax *node) { return loco::dtype_get(node->input()); }
  loco::DataType visit(const loco::TensorTranspose *node) { return loco::dtype_get(node->input()); }
  loco::DataType visit(const loco::TransposedConv2D *node) { return loco::dtype_get(node->ifm()); }
};

} // namespace

namespace loco
{

bool CanonicalTypeInferenceRule::recognize(const Dialect *d) const
{
  // This rule recognizes only "loco.canonical" dialect!
  return CanonicalDialect::get() == d;
}

bool CanonicalTypeInferenceRule::infer(const Node *node, DataType &dtype) const
{
  assert(node->dialect() == loco::CanonicalDialect::get());
  assert(dynamic_cast<const loco::CanonicalNode *>(node) != nullptr);

  CanonicalTypeForwardAlgorithm alg;
  dtype = loco::must_cast<const loco::CanonicalNode *>(node)->accept(&alg);

  return true;
}

bool MultiDialectTypeInferenceRule::recognize(const Dialect *d) const
{
  const auto found = _rules.find(d);

  if (found == _rules.cend())
    return false;

  auto rule = found->second;
  auto result = rule->recognize(d);

  return result;
}

bool MultiDialectTypeInferenceRule::infer(const Node *node, DataType &dtype) const
{
  const auto found = _rules.find(node->dialect());

  if (found == _rules.cend())
    return false;

  auto rule = found->second;
  if (rule->infer(node, dtype))
    return true;

  return false;
}

MultiDialectTypeInferenceRule &MultiDialectTypeInferenceRule::bind(const Dialect *d,
                                                                   const TypeInferenceRule *rule)
{
  assert(_rules.find(d) == _rules.end());
  assert(rule->recognize(d));

  _rules[d] = rule;

  return (*this);
}

} // namespace loco
