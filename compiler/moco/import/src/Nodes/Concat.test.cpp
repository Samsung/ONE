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

#include "moco/Import/Nodes/Concat.h"
#include "TestHelper.h"

#include <gtest/gtest.h>

using namespace moco::test;

namespace
{

// clang-format off
const char *concat_01_pbtxtdata = STRING_CONTENT(
  name: "Concat"
  op: "ConcatV2"
  input: "Input01"
  input: "Input02"
  input: "Axis"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
);
// clang-format on

} // namespace

TEST(TensorFlowImport, concat_01)
{
  TFNodeBuildTester tester;
  moco::ConcatV2GraphBuilder graphbuilder;
  tensorflow::NodeDef nodedef;

  EXPECT_TRUE(plier::tf::parse_nodedef(concat_01_pbtxtdata, nodedef));

  // what to test:
  // - there should exist TFConcatV2
  // - there should be two values
  // - values(idx) should not be nullptr
  // - axis() should not be nullptr

  tester.inputs({"Input01", "Input02", "Axis"});
  tester.output("Concat");
  tester.run(nodedef, graphbuilder);

  auto test_node = dynamic_cast<moco::TFConcatV2 *>(tester.output());
  ASSERT_NE(test_node, nullptr);
  ASSERT_EQ(test_node->num_values(), 2);
}

namespace
{

// clang-format off
const char *concat_02_pbtxtdata = STRING_CONTENT(
  name: "Concat"
  op: "ConcatV2"
  input: "Input01"
  input: "Input02"
  input: "Input03"
  input: "Axis"
  attr {
    key: "N"
    value {
      i: 3
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
);
// clang-format on

} // namespace

TEST(TensorFlowImport, concat_02)
{
  TFNodeBuildTester tester;
  moco::ConcatV2GraphBuilder graphbuilder;
  tensorflow::NodeDef nodedef;

  EXPECT_TRUE(plier::tf::parse_nodedef(concat_02_pbtxtdata, nodedef));

  // what to test: TFConcatV2 has 3 inputs
  // - there should exist TFConcatV2
  // - values(idx) should not be nullptr
  // - axis() should not be nullptr

  tester.inputs({"Input01", "Input02", "Input03", "Axis"});
  tester.output("Concat");
  tester.run(nodedef, graphbuilder);

  auto test_node = dynamic_cast<moco::TFConcatV2 *>(tester.output());
  ASSERT_NE(test_node, nullptr);
  ASSERT_EQ(test_node->num_values(), 3);
}
