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

#include "moco/Import/Nodes/FusedBatchNorm.h"
#include "TestHelper.h"

#include <gtest/gtest.h>

using namespace moco::test;

namespace
{
// clang-format off
const char *fbn_basic_pbtxt = STRING_CONTENT(
  name: "FBN_01"
  op: "FusedBatchNorm"
  input: "input"
  input: "gamma"
  input: "beta"
  input: "FBN_01/mean"
  input: "FBN_01/variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "epsilon"
    value {
      f: 0.001
    }
  }
  attr {
    key: "is_training"
    value {
      b: false
    }
  }
);
// clang-format on

} // namespace

TEST(TensorFlowImport, tf_fbn_basic)
{
  TFNodeBuildTester tester;
  moco::FusedBatchNormGraphBuilder graphbuilder;
  tensorflow::NodeDef nodedef;

  EXPECT_TRUE(plier::tf::parse_nodedef(fbn_basic_pbtxt, nodedef));

  // what to test:
  // - there should exist a TFFusedBatchNorm
  // - input() should not be nullptr
  // - gamma() should not be nullptr
  // - beta() should not be nullptr
  // - mean() should not be nullptr
  // - variance() should not be nullptr
  // - epsilon() value should match

  tester.inputs({"input", "gamma", "beta", "FBN_01/mean", "FBN_01/variance"});
  tester.output("FBN_01");
  tester.run(nodedef, graphbuilder);

  auto test_node = dynamic_cast<moco::TFFusedBatchNorm *>(tester.output());
  ASSERT_NE(test_node, nullptr);
  ASSERT_EQ(test_node->epsilon(), 0.001f);
}
