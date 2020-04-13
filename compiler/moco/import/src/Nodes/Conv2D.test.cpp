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

#include "moco/Import/Nodes/Conv2D.h"
#include "TestHelper.h"

#include <gtest/gtest.h>

using namespace moco::test;

namespace
{
// clang-format off
const char *conv2d_01_pbtxtdata = STRING_CONTENT(
  name: "conv2d"
  op: "Conv2D"
  input: "ifm"
  input: "ker"
  attr { key: "T" value { type: DT_FLOAT } }
  attr { key: "data_format"  value { s: "NHWC" } }
  attr { key: "dilations" value { list { i: 1 i: 1 i: 1 i: 1 } } }
  attr { key: "padding" value { s: "VALID" } }
  attr { key: "strides" value { list { i: 1 i: 2 i: 3 i: 1 } } }
);
// clang-format on
} // namespace

TEST(TensorFlowImport, Conv2D_01)
{
  TFNodeBuildTester tester;
  moco::Conv2DGraphBuilder graphbuilder;
  tensorflow::NodeDef nodedef;

  EXPECT_TRUE(plier::tf::parse_nodedef(conv2d_01_pbtxtdata, nodedef));

  // what to test:
  // - Conv2D node should exist
  // - ifm() should not be nullptr
  // - ker() should not be nullptr
  // - attribute values should match

  tester.inputs({"ifm", "ker"});
  tester.output("conv2d");
  tester.run(nodedef, graphbuilder);

  auto test_node = dynamic_cast<moco::TFConv2D *>(tester.output());
  ASSERT_NE(test_node, nullptr);
  ASSERT_EQ(test_node->padding(), "VALID");
  ASSERT_EQ(test_node->data_layout(), "NHWC");
  auto strides = test_node->strides();
  ASSERT_EQ(strides.size(), 4);
  // TODO add verify dilation
}

namespace
{
// clang-format off
const char *conv2d_inception_pbtxtdata = STRING_CONTENT(
  name: "InceptionV3/InceptionV3/Conv2d_1a_3x3/Conv2D"
  op: "Conv2D"
  input: "input:0"
  input: "InceptionV3/Conv2d_1a_3x3/weights/read/_3__cf__3"
  attr {
    key: "T"
    value { type: DT_FLOAT }
  }
  attr {
    key: "data_format"
    value { s: "NHWC" }
  }
  attr {
    key: "dilations"
    value {
      list { i: 1 i: 1 i: 1 i: 1 }
    }
  }
  attr {
    key: "padding"
    value { s: "VALID" }
  }
  attr {
    key: "strides"
    value {
      list { i: 1 i: 2 i: 2 i: 1 }
    }
  }
);
} // namespace

TEST(TensorFlowImport, Conv2D_inception_indexed_tensor_name)
{
  TFNodeBuildTester tester;
  moco::Conv2DGraphBuilder graphbuilder;
  tensorflow::NodeDef nodedef;

  EXPECT_TRUE(plier::tf::parse_nodedef(conv2d_inception_pbtxtdata, nodedef));

  // what to test: name with ':0' should be treated correctly
  // - Conv2D node should exist
  // - ifm() should not be nullptr
  // - ker() should not be nullptr

  tester.inputs({"input", "InceptionV3/Conv2d_1a_3x3/weights/read/_3__cf__3"});
  tester.output("InceptionV3/InceptionV3/Conv2d_1a_3x3/Conv2D");
  tester.run(nodedef, graphbuilder);
}
