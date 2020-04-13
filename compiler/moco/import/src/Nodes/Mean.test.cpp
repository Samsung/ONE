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

#include "moco/Import/Nodes/Mean.h"
#include "TestHelper.h"

#include <gtest/gtest.h>

using namespace moco::test;

namespace
{

// clang-format off
const char *mean_true_pbtxtdata = STRING_CONTENT(
  name: "Mean"
  op: "Mean"
  input: "Placeholder"
  input: "Const"
  attr {
    key: "T"
    value { type: DT_FLOAT }
  }
  attr {
    key: "Tidx"
    value { type: DT_INT32 }
  }
  attr {
    key: "keep_dims"
    value { b: true }
  }
);
// clang-format on

} // namespace

TEST(TensorFlowImport, mean_true)
{
  TFNodeBuildTester tester;
  moco::MeanGraphBuilder graphbuilder;
  tensorflow::NodeDef nodedef;

  EXPECT_TRUE(plier::tf::parse_nodedef(mean_true_pbtxtdata, nodedef));

  // what to test:
  // - there should exist TFMean
  // - input node should not be nullptr
  // - reduction_indeces node should not be nullptr
  // - keep_dims attribute is set same as pbtxt

  tester.inputs({"Placeholder", "Const"});
  tester.output("Mean");
  tester.run(nodedef, graphbuilder);

  auto test_node = dynamic_cast<moco::TFMean *>(tester.output());
  ASSERT_NE(test_node, nullptr);
  ASSERT_EQ(test_node->keep_dims(), true);
}

namespace
{

// clang-format off
const char *mean_false_pbtxtdata = STRING_CONTENT(
  name: "Mean"
  op: "Mean"
  input: "Placeholder"
  input: "Const"
  attr {
    key: "T"
    value { type: DT_FLOAT }
  }
  attr {
    key: "Tidx"
    value { type: DT_INT32 }
  }
  attr {
    key: "keep_dims"
    value { b: false }
  }
);
// clang-format on

} // namespace

TEST(TensorFlowImport, mean_false)
{
  TFNodeBuildTester tester;
  moco::MeanGraphBuilder graphbuilder;
  tensorflow::NodeDef nodedef;

  EXPECT_TRUE(plier::tf::parse_nodedef(mean_false_pbtxtdata, nodedef));

  // what to test:
  // - there should exist TFMean
  // - input node should not be nullptr
  // - reduction_indeces node should not be nullptr
  // - keep_dims attribute is set same as pbtxt

  tester.inputs({"Placeholder", "Const"});
  tester.output("Mean");
  tester.run(nodedef, graphbuilder);

  auto test_node = dynamic_cast<moco::TFMean *>(tester.output());
  ASSERT_NE(test_node, nullptr);
  ASSERT_EQ(test_node->keep_dims(), false);
}
