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

#include "moco/Import/Nodes/Pad.h"
#include "TestHelper.h"

#include <gtest/gtest.h>

using namespace moco::test;

namespace
{
// clang-format off
const char *pad_basic_pbtxt = STRING_CONTENT(
  name: "Pad"
  op: "Pad"
  input: "input"
  input: "paddings"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tpaddings"
    value {
      type: DT_INT32
    }
  }
);
// clang-format on

} // namespace

TEST(TensorFlowImport, tf_pad_basic)
{
  TFNodeBuildTester tester;
  moco::PadGraphBuilder graphbuilder;
  tensorflow::NodeDef nodedef;

  EXPECT_TRUE(plier::tf::parse_nodedef(pad_basic_pbtxt, nodedef));

  // what to test:
  // - TFPad node should exist
  // - input input() should not be null
  // - input paddings() should not be null

  tester.inputs({"input", "paddings"});
  tester.output("Pad");
  tester.run(nodedef, graphbuilder);
}
