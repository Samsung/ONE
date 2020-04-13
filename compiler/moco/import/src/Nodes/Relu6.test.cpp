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

#include "moco/Import/Nodes/Relu6.h"
#include "TestHelper.h"

#include <gtest/gtest.h>

using namespace moco::test;

namespace
{

// clang-format off
const char *relu6_01_pbtxtdata = STRING_CONTENT(
  name: "ReLU6"
  op: "Relu6"
  input: "Placeholder"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
);
// clang-format on

} // namespace

TEST(TensorFlowImport, relu6_01)
{
  TFNodeBuildTester tester;
  moco::Relu6GraphBuilder graphbuilder;
  tensorflow::NodeDef nodedef;

  EXPECT_TRUE(plier::tf::parse_nodedef(relu6_01_pbtxtdata, nodedef));

  // what to test:
  // - there should exist TFRelu6
  // - features node should not be null

  tester.inputs({"Placeholder"});
  tester.output("ReLU6");
  tester.run(nodedef, graphbuilder);
}
