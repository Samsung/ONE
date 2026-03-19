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

#include "moco/Import/Nodes/Placeholder.h"
#include "TestHelper.h"

#include <gtest/gtest.h>

using namespace moco::test;

namespace
{
// clang-format off
const char *known_batch_pbtxt = STRING_CONTENT(
  name: "placeholder"
  op: "Placeholder"
  attr {
    key: "dtype" value { type: DT_FLOAT }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim { size: 1024 }
        dim { size: 2 }
        dim { size: 3 }
        dim { size: 4 }
      }
    }
  }
);
// clang-format on

} // namespace

TEST(TensorFlowImport, placeholder_knwon_batch)
{
  TFNodeBuildTester tester;
  moco::PlaceholderGraphBuilder graphbuilder;
  tensorflow::NodeDef nodedef;

  EXPECT_TRUE(plier::tf::parse_nodedef(known_batch_pbtxt, nodedef));

  // what to test:
  // - TFPlaceholder node should exist
  // - shape attribute should match

  tester.inputs({});
  tester.output("placeholder");
  tester.run(nodedef, graphbuilder);

  auto test_node = dynamic_cast<moco::TFPlaceholder *>(tester.output());
  assert(test_node != nullptr);
  ASSERT_TRUE(test_node->dim(0).known() && test_node->dim(0).value() == 1024);
  ASSERT_TRUE(test_node->dim(1).known() && test_node->dim(1).value() == 2);
  ASSERT_TRUE(test_node->dim(2).known() && test_node->dim(2).value() == 3);
  ASSERT_TRUE(test_node->dim(3).known() && test_node->dim(3).value() == 4);
}
