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

#include "moco/Import/Nodes/AvgPool.h"
#include "TestHelper.h"

#include <gtest/gtest.h>

using namespace moco::test;

namespace
{
// clang-format off
const char *avgpool_01_pbtxtdata = STRING_CONTENT(
  name: "avgpool"
  op: "AvgPool"
  input: "const/float"
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
    key: "ksize"
    value {
      list {
        i: 1
        i: 2
        i: 3
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
        i: 3
        i: 2
        i: 1
      }
    }
  }
);
// clang-format on

} // namespace

TEST(TensorFlowImport, AvgPool_01)
{
  TFNodeBuildTester tester;
  moco::AvgPoolGraphBuilder graphbuilder;
  tensorflow::NodeDef nodedef;

  EXPECT_TRUE(plier::tf::parse_nodedef(avgpool_01_pbtxtdata, nodedef));

  // what to test:
  // - there should exist TFAvgPool
  // - input should exist
  // - attributes value should match

  tester.inputs({"const/float"});
  tester.output("avgpool");
  tester.run(nodedef, graphbuilder);

  auto test_node = dynamic_cast<moco::TFAvgPool *>(tester.output());
  ASSERT_NE(test_node, nullptr);
  ASSERT_EQ(test_node->data_layout(), "NHWC");
  ASSERT_EQ(test_node->padding(), "VALID");
  ASSERT_EQ(test_node->ksize(), std::vector<int64_t>({1, 2, 3, 1}));
  ASSERT_EQ(test_node->strides(), std::vector<int64_t>({1, 3, 2, 1}));
}
