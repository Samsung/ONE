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

#include "moco/Import/Nodes/StridedSlice.h"
#include "TestHelper.h"

#include <gtest/gtest.h>

using namespace moco::test;

namespace
{
// clang-format off
const char *stridedslice_basic_pbtxt = STRING_CONTENT(
  name: "StridedSlice"
  op: "StridedSlice"
  input: "input"
  input: "begin"
  input: "end"
  input: "strides"
  attr {
    key: "Index"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "begin_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "ellipsis_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "end_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "new_axis_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "shrink_axis_mask"
    value {
      i: 1
    }
  }
);
// clang-format on

} // namespace

TEST(TensorFlowImport, tf_stridedslice_basic)
{
  TFNodeBuildTester tester;
  moco::StridedSliceGraphBuilder graphbuilder;
  tensorflow::NodeDef nodedef;

  EXPECT_TRUE(plier::tf::parse_nodedef(stridedslice_basic_pbtxt, nodedef));

  // what to test:
  // - TFStridedSlice node should exist
  // - inputs should not be nullptr
  // - attributes should match the values

  tester.inputs({"input", "begin", "end", "strides"}, loco::DataType::S32);
  tester.output("StridedSlice");
  tester.run(nodedef, graphbuilder);

  auto test_node = dynamic_cast<moco::TFStridedSlice *>(tester.output());
  ASSERT_NE(test_node, nullptr);
  ASSERT_EQ(test_node->begin_mask(), 0);
  ASSERT_EQ(test_node->end_mask(), 0);
  ASSERT_EQ(test_node->ellipsis_mask(), 0);
  ASSERT_EQ(test_node->new_axis_mask(), 0);
  ASSERT_EQ(test_node->shrink_axis_mask(), 1);
}

// TODO add test where strides is None
