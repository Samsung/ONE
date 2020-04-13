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

#include "moco/Import/Nodes/BiasAdd.h"
#include "TestHelper.h"

#include <gtest/gtest.h>

using namespace moco::test;

namespace
{

// clang-format off
const char *bias_add_01_pbtxtdata = STRING_CONTENT(
  name: "out"
  op: "BiasAdd"
  input: "val"
  input: "bias"
  attr {
    key: "T"
    value { type: DT_FLOAT }
  }
  attr {
    key: "data_format"
    value { s: "NHWC" }
  }
);
// clang-format on

} // namespace

TEST(TensorFlowImport, bias_add_01)
{
  TFNodeBuildTester tester;
  moco::BiasAddGraphBuilder graphbuilder;
  tensorflow::NodeDef nodedef;

  EXPECT_TRUE(plier::tf::parse_nodedef(bias_add_01_pbtxtdata, nodedef));

  // what to test:
  // - there should exist TFBiasAdd
  // - value() should not be nullptr
  // - bias() should not be nullptr
  // - data_layout should match

  tester.inputs({"val", "bias"});
  tester.output("out");
  tester.run(nodedef, graphbuilder);

  auto test_node = dynamic_cast<moco::TFBiasAdd *>(tester.output());
  ASSERT_NE(test_node, nullptr);
  ASSERT_TRUE(test_node->data_layout() == "NHWC");
}

namespace
{

// clang-format off
const char *bias_add_NCHW_pbtxtdata = STRING_CONTENT(
  name: "out"
  op: "BiasAdd"
  input: "val"
  input: "bias"
  attr {
    key: "T"
    value { type: DT_FLOAT }
  }
  attr {
    key: "data_format"
    value { s: "NCHW" }
  }
);
// clang-format on

} // namespace

TEST(TensorFlowImport, bias_add_NCHW_axis)
{
  TFNodeBuildTester tester;
  moco::BiasAddGraphBuilder graphbuilder;
  tensorflow::NodeDef nodedef;

  EXPECT_TRUE(plier::tf::parse_nodedef(bias_add_NCHW_pbtxtdata, nodedef));

  // what to test:
  // - there should exist TFBiasAdd
  // - value() should not be nullptr
  // - bias() should not be nullptr
  // - data_layout should match

  tester.inputs({"val", "bias"});
  tester.output("out");
  tester.run(nodedef, graphbuilder);

  auto test_node = dynamic_cast<moco::TFBiasAdd *>(tester.output());
  ASSERT_NE(test_node, nullptr);
  ASSERT_TRUE(test_node->data_layout() == "NCHW");
}
