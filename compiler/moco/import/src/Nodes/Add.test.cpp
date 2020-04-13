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

#include "moco/Import/Nodes/Add.h"
#include "TestHelper.h"

#include <gtest/gtest.h>

using namespace moco::test;

namespace
{
// clang-format off
const char *add_basic_pbtxt = STRING_CONTENT(
  name: "ADD_01"
  op: "Add"
  input: "input_01"
  input: "input_02"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
);
// clang-format on

} // namespace

TEST(TensorFlowImport, tf_add_basic)
{
  TFNodeBuildTester tester;
  moco::AddGraphBuilder graphbuilder;
  tensorflow::NodeDef nodedef;

  EXPECT_TRUE(plier::tf::parse_nodedef(add_basic_pbtxt, nodedef));

  // what to test:
  // - TFAdd node should exist
  // - both inputs x() and y() should not be null

  tester.inputs({"input_01", "input_02"});
  tester.output("ADD_01");
  tester.run(nodedef, graphbuilder);
}
