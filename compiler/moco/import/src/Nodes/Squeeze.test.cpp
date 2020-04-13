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

#include "moco/Import/Nodes/Squeeze.h"
#include "TestHelper.h"

#include <gtest/gtest.h>

using namespace moco::test;

namespace
{

// clang-format off
const char *squeeze_all_pbtxtdata = STRING_CONTENT(
  name: "Squeeze"
  op: "Squeeze"
  input: "Placeholder"
  attr {
    key: "T"
    value { type: DT_FLOAT }
  }
);
// clang-format on

} // namespace

TEST(TensorFlowImport, squeeze_all)
{
  TFNodeBuildTester tester;
  moco::SqueezeGraphBuilder graphbuilder;
  tensorflow::NodeDef nodedef;

  EXPECT_TRUE(plier::tf::parse_nodedef(squeeze_all_pbtxtdata, nodedef));

  // what to test:
  // - there should exist TFSqueeze
  // - input node should not be null
  // - squeeze_dims attribute is set same as pbtxt

  tester.inputs({"Placeholder"});
  tester.output("Squeeze");
  tester.run(nodedef, graphbuilder);

  auto test_node = dynamic_cast<moco::TFSqueeze *>(tester.output());
  ASSERT_NE(test_node, nullptr);
  ASSERT_EQ(test_node->squeeze_dims().size(), 0);
}

namespace
{

// clang-format off
const char *squeeze_some_pbtxtdata = STRING_CONTENT(
  name: "Squeeze"
  op: "Squeeze"
  input: "Placeholder"
  attr {
    key: "T"
    value { type: DT_FLOAT }
  }
  attr {
    key: "squeeze_dims"
    value {
      list { i: 1 }
    }
  }
);
// clang-format on

} // namespace

TEST(TensorFlowImport, squeeze_some)
{
  TFNodeBuildTester tester;
  moco::SqueezeGraphBuilder graphbuilder;
  tensorflow::NodeDef nodedef;

  EXPECT_TRUE(plier::tf::parse_nodedef(squeeze_some_pbtxtdata, nodedef));

  // what to test:
  // - there should exist TFSqueeze
  // - input node should not be null
  // - squeeze_dims attribute is set same as pbtxt

  tester.inputs({"Placeholder"});
  tester.output("Squeeze");
  tester.run(nodedef, graphbuilder);

  auto test_node = dynamic_cast<moco::TFSqueeze *>(tester.output());
  ASSERT_NE(test_node, nullptr);
  ASSERT_EQ(test_node->squeeze_dims().size(), 1);
  ASSERT_EQ(test_node->squeeze_dims().at(0), 1);
}

// TODO Add test case for negative squeeze dim
