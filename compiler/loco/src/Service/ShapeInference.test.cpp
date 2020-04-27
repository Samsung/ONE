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

#include "loco/Service/ShapeInference.h"
#include "GraphTestcase.h"

#include <vector>

#include <gtest/gtest.h>

// This test validates whether framework works as expected.
TEST(ShapeInferenceTest, framework)
{
  // Mock-up Shape Inference Rule
  struct SampleShapeInferenceRule final : public loco::ShapeInferenceRule
  {
  public:
    SampleShapeInferenceRule(std::vector<const loco::Node *> *nodes) : _nodes{nodes}
    {
      // DO NOTHING
    }

  public:
    // Accept all the dialects
    bool recognize(const loco::Dialect *) const final { return true; }

    bool infer(const loco::Node *node, loco::NodeShape &shape) const final
    {
      // Record the order of inference
      _nodes->emplace_back(node);

      if (_nodes->size() != 1)
      {
        return false;
      }

      // Set the first node as Tensor<1>
      loco::TensorShape tensor_shape;

      tensor_shape.rank(1);
      tensor_shape.dim(0) = 4;

      shape.set(tensor_shape);

      return true;
    }

  private:
    std::vector<const loco::Node *> *_nodes;
  };

  GraphTestcase<GraphCode::Identity> testcase;

  std::vector<const loco::Node *> nodes;

  SampleShapeInferenceRule rule{&nodes};

  loco::apply(&rule).to(testcase.graph());

  // Framework SHOULD visit all the nodes
  ASSERT_EQ(2, nodes.size());
  // Framework SHOULD visit "pull" before "push"
  ASSERT_EQ(testcase.pull_node, nodes.at(0));
  ASSERT_EQ(testcase.push_node, nodes.at(1));

  // Framework SHOULD make an annotation if "rule" returns TRUE
  ASSERT_TRUE(loco::shape_known(testcase.pull_node));
  ASSERT_EQ(loco::Domain::Tensor, loco::shape_get(testcase.pull_node).domain());
  ASSERT_EQ(1, loco::shape_get(testcase.pull_node).as<loco::TensorShape>().rank());
  ASSERT_EQ(4, loco::shape_get(testcase.pull_node).as<loco::TensorShape>().dim(0));

  // Framework SHOULD NOT make any annotation if "rule" returns FALSE
  ASSERT_FALSE(loco::shape_known(testcase.push_node));
}
