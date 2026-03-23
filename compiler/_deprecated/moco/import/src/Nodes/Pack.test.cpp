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

#include "moco/Import/Nodes/Pack.h"
#include "TestHelper.h"

#include <gtest/gtest.h>

using namespace moco::test;

namespace
{

// clang-format off
const char *pack_01_pbtxtdata = STRING_CONTENT(
  name: "Pack"
  op: "Pack"
  input: "input_1"
  input: "input_2"
  input: "input_3"
  input: "input_4"
  attr {
    key: "N"
    value {
      i: 4
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "axis"
    value {
      i: 0
    }
  }
);
// clang-format on

} // namespace

TEST(TensorFlowImport, tf_pack_basic)
{
  TFNodeBuildTester tester;
  moco::PackGraphBuilder graphbuilder;
  tensorflow::NodeDef nodedef;

  EXPECT_TRUE(plier::tf::parse_nodedef(pack_01_pbtxtdata, nodedef));

  // what to test:
  // - there should exist TFPack
  // - there should be four values
  // - values(idx) should not be nullptr
  // - axis() should be 0

  tester.inputs({"input_1", "input_2", "input_3", "input_4"});
  tester.output("Pack");
  tester.run(nodedef, graphbuilder);

  auto test_node = loco::must_cast<moco::TFPack *>(tester.output());
  ASSERT_NE(test_node, nullptr);
  ASSERT_EQ(test_node->N(), 4);
  ASSERT_NE(test_node->values(0), nullptr);
  ASSERT_NE(test_node->values(1), nullptr);
  ASSERT_NE(test_node->values(2), nullptr);
  ASSERT_NE(test_node->values(3), nullptr);
  ASSERT_EQ(test_node->axis(), 0);
}
