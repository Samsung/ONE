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

#include "moco/Import/Nodes/DepthwiseConv2dNative.h"
#include "TestHelper.h"

#include <gtest/gtest.h>

using namespace moco::test;

namespace
{
// clang-format off
const char *depthwise_conv2d_native_01_pbtxtdata = STRING_CONTENT(
  name: "depthwise"
  op: "DepthwiseConv2dNative"
  input: "input"
  input: "filter"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "VALID"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
);
// clang-format on
} // namespace

TEST(TensorFlowImport, Depthwise_conv2d_native)
{
  TFNodeBuildTester tester;
  moco::DepthwiseConv2dNativeGraphBuilder graphbuilder;
  tensorflow::NodeDef nodedef;

  EXPECT_TRUE(plier::tf::parse_nodedef(depthwise_conv2d_native_01_pbtxtdata, nodedef));

  // what to test:
  // - All node inputs are valid
  // - All attributes are as expected

  tester.inputs({"input", "filter"});
  tester.output("depthwise");
  tester.run(nodedef, graphbuilder);

  auto test_node = dynamic_cast<moco::TFDepthwiseConv2dNative *>(tester.output());
  ASSERT_NE(test_node, nullptr);
  ASSERT_EQ(test_node->padding(), "VALID");
  ASSERT_EQ(test_node->data_layout(), "NHWC");
  ASSERT_EQ(test_node->strides().size(), 4);
}
