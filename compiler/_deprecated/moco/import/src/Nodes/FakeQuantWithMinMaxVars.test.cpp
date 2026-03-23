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

#include "moco/Import/Nodes/FakeQuantWithMinMaxVars.h"
#include "TestHelper.h"

#include <gtest/gtest.h>

using namespace moco::test;

namespace
{
// clang-format off
const char *fakequant_01_pbtxtdata = STRING_CONTENT(
  name: "FakeQuant"
  op: "FakeQuantWithMinMaxVars"
  input: "Input"
  input: "FakeMin"
  input: "FakeMax"
  attr {
    key: "narrow_range"
    value { b: true  }
  }
  attr {
    key: "num_bits"
    value { i: 16 }
  }
);
// clang-format on
} // namespace

TEST(TensorFlowImport, FakeQuantWithMinMaxVars)
{
  TFNodeBuildTester tester;
  moco::FakeQuantWithMinMaxVarsGraphBuilder graphbuilder;
  tensorflow::NodeDef nodedef;

  EXPECT_TRUE(plier::tf::parse_nodedef(fakequant_01_pbtxtdata, nodedef));

  // what to test:
  // - All node inputs are valid
  // - All attributes are as expected

  tester.inputs({"Input", "FakeMin", "FakeMax"});
  tester.output("FakeQuant");
  tester.run(nodedef, graphbuilder);

  auto test_node = dynamic_cast<moco::TFFakeQuantWithMinMaxVars *>(tester.output());
  ASSERT_NE(test_node, nullptr);
  ASSERT_EQ(test_node->narrow_range(), true);
  ASSERT_EQ(test_node->num_bits(), 16);
}
