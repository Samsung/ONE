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

#include "moco/Import/Nodes/Shape.h"
#include "TestHelper.h"

#include <gtest/gtest.h>

using namespace moco::test;

namespace
{

// clang-format off
const char *shape_000_pbtxtdata = STRING_CONTENT(
  name: "Shape"
  op: "Shape"
  input: "Placeholder"
  attr {
    key: "T"
    value { type: DT_FLOAT }
  }
  attr {
    key: "out_type"
    value { type: DT_INT32 }
  }
);
// clang-format on

} // namespace

TEST(TensorFlowImport, shape_000)
{
  TFNodeBuildTester tester;
  moco::ShapeGraphBuilder graphbuilder;
  tensorflow::NodeDef nodedef;

  EXPECT_TRUE(plier::tf::parse_nodedef(shape_000_pbtxtdata, nodedef));

  // what to test:
  // - there should exist TFShape
  // - input node should not be null
  // - dtype attribute is set same as out_type attribute of pbtxt

  tester.inputs({"Placeholder"});
  tester.output("Shape");
  tester.run(nodedef, graphbuilder);

  auto test_node = dynamic_cast<moco::TFShape *>(tester.output());
  ASSERT_NE(test_node, nullptr);
  ASSERT_EQ(test_node->dtype(), loco::DataType::S32);
}
